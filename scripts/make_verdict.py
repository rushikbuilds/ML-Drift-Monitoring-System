"""Day 6: compute the v6 verdict mechanically from saved results.

    python scripts/make_verdict.py

Applies the PASS rules frozen in dews/v6_config.py / docs/v6_protocol.md section 7 to
the one-time held-out results (artifacts/v6/day5_held_out.json) and to the natural-stream
windows (H5). Nothing is fitted; the frozen calibrators and thresholds are verified by
fingerprint first. Writes artifacts/v6/verdict.json and docs/verdict.md.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dews import v2  # noqa: E402
from dews import v6_config as cfg  # noqa: E402
from dews import v6_eval as ev  # noqa: E402
from scripts.run_v6_develop import setting_key  # noqa: E402
from scripts.run_v6_extract import load_signals  # noqa: E402
from scripts.run_v6_heldout import check_frozen  # noqa: E402

OUT = ROOT / "artifacts" / "v6"
B = 2000
RNG = np.random.default_rng(20261108)
PRIMARY = setting_key(cfg.PRIMARY["model"], cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"])


def boot_mean(x):
    x = np.asarray(x, float)
    means = [RNG.choice(x, len(x)).mean() for _ in range(B)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_auroc(pos, neg):
    pos, neg, vals = np.asarray(pos), np.asarray(neg), []
    for _ in range(B):
        p, n = RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))
        vals.append(roc_auc_score([1] * len(p) + [0] * len(n), np.concatenate([p, n])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def h5_natural(calibrator, thresholds):
    files = sorted(OUT.glob("natural_signals_*of*.pkl"))
    windows = [w for f in files for w in pickle.loads(f.read_bytes())]
    delay, H = cfg.PRIMARY["label_delay"], cfg.HORIZON_BATCHES
    sys_hits, lab_hits, fa = [], [], {"primary_system": [0, 0], "label_trend": [0, 0]}
    for w in windows:
        n = len(w["accuracy"])
        scores = {"primary_system": calibrator.score(w),
                  "label_trend": ev.baseline_scores(w, delay, 1.0)["label_trend"]}
        event = v2.outcome(np.asarray(w["accuracy"]))["event_onset"]
        t = np.arange(n)
        clean = (t >= cfg.WARMUP_BATCHES) & ((t < event - H) if event is not None else True)
        hits = {}
        for m, s in scores.items():
            alerts = (s >= thresholds[m]) & (t >= cfg.WARMUP_BATCHES)
            fa[m][0] += int((alerts & clean).sum())
            fa[m][1] += int(clean.sum())
            if event is not None:
                win = (t >= max(event - H, cfg.WARMUP_BATCHES)) & (t < cfg.confirmation_batch(event, delay))
                hits[m] = bool((alerts & win).any())
        if event is not None:
            sys_hits.append(hits["primary_system"])
            lab_hits.append(hits["label_trend"])
    s, l = np.array(sys_hits, float), np.array(lab_hits, float)
    diffs = [(RNG.choice(np.arange(len(s)), len(s))) for _ in range(B)]
    ci = [float(np.percentile([s[i].mean() - l[i].mean() for i in diffs], q)) for q in (2.5, 97.5)]
    return {"windows": len(windows), "event_windows": len(s), "datasets": len({w["dataset"] for w in windows}),
            "t2_system": float(s.mean()), "t2_label_trend": float(l.mean()),
            "difference": float(s.mean() - l.mean()), "ci95": ci,
            "false_alerts_per_100_clean": {m: 100.0 * a / max(b, 1) for m, (a, b) in fa.items()},
            "t2_window": "alerts in [max(event - 5, 6), event + 1 + max(L, 1)); intervention onset is unknown on natural streams"}


def main() -> None:
    check_frozen()
    held = json.loads((OUT / "day5_held_out.json").read_text())["results"]
    thresholds = json.loads((OUT / "frozen_choices.json").read_text())["thresholds"]
    calibrator = pickle.loads((OUT / "frozen_calibrators.pkl").read_bytes())[PRIMARY]
    r = held[PRIMARY]
    sysr, lab, fixed = r["primary_system"], r["label_trend"], r["fixed_dews_score"]
    hyp = {}

    # H1
    hits = sysr["gradual"]["t1_hits"]
    share, lead, fa = float(np.mean(hits)), sysr["gradual"]["t1_median_lead"], sysr["false_alerts_per_100_clean"]
    hyp["H1"] = {"value": share, "ci95": boot_mean(hits), "median_lead": lead, "false_alerts": fa,
                 "episodes": len(hits), "target": "share >= 0.60, median lead >= 2, false alerts <= 5",
                 "pass": bool(share >= 0.60 and (lead or 0) >= 2 and fa <= cfg.FALSE_ALERT_BUDGET)}

    # H2 (paired over the same harmful episodes)
    s, l = np.array(sysr["harmful"]["t2_hits"], float), np.array(lab["harmful"]["t2_hits"], float)
    idx = [RNG.choice(len(s), len(s)) for _ in range(B)]
    ci = [float(np.percentile([s[i].mean() - l[i].mean() for i in idx], q)) for q in (2.5, 97.5)]
    diff = float(s.mean() - l.mean())
    hyp["H2"] = {"value": diff, "ci95": ci, "t2_system": float(s.mean()), "t2_label_trend": float(l.mean()),
                 "episodes": len(s), "target": "difference >= 0.10 and CI lower bound > 0",
                 "pass": bool(diff >= 0.10 and ci[0] > 0)}

    # H3 (recomputed from held-out signals with the frozen calibrator; deterministic)
    episodes = load_signals("held_out", cfg.PRIMARY["model"])
    scores = ev.score_matrix(episodes, cfg.PRIMARY["label_delay"], cfg.PRIMARY["label_fraction"],
                             lambda e: calibrator)["primary_system"]
    masks = ev.build_masks(episodes, cfg.PRIMARY["label_delay"])
    harm = lambda i: float(np.nanmean(scores[i, masks.onset[i]:masks.onset[i] + ev.HARM_WINDOW]))  # noqa: E731
    pos = [harm(i) for i in np.flatnonzero(masks.harmful_gradual | masks.harmful_abrupt)]
    neg = [harm(i) for i in np.flatnonzero(masks.harmless_shift)]
    auc = float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))
    alerts = np.where(np.isfinite(scores), scores >= thresholds[PRIMARY]["primary_system"], False)
    harmless_hits = (alerts & masks.after_onset)[np.flatnonzero(masks.harmless_shift)].any(axis=1).astype(float)
    hyp["H3"] = {"value": auc, "ci95": boot_auroc(pos, neg), "harmless_alert_share": float(harmless_hits.mean()),
                 "harmless_alert_ci95": boot_mean(harmless_hits), "harmful": len(pos), "harmless": len(neg),
                 "target": "AUROC >= 0.80 and harmless alert share <= 0.10",
                 "pass": bool(auc >= 0.80 and harmless_hits.mean() <= 0.10)}

    # H4
    wins, rows = 0, {}
    for d, a in sysr["per_dataset"].items():
        b = fixed["per_dataset"][d]
        win = (a["t1_share_harmful"] or 0) > (b["t1_share_harmful"] or 0) and (a["t2_share_harmful"] or 0) > (b["t2_share_harmful"] or 0)
        wins += win
        rows[d] = {"system_t1_t2": [a["t1_share_harmful"], a["t2_share_harmful"]],
                   "fixed_t1_t2": [b["t1_share_harmful"], b["t2_share_harmful"]], "system_wins": bool(win)}
    hyp["H4"] = {"value": wins, "datasets": len(rows), "per_dataset": rows,
                 "fixed_score_false_alerts": fixed["false_alerts_per_100_clean"], "system_false_alerts": fa,
                 "target": ">= 4 of 5 held-out datasets", "pass": bool(wins >= 4)}

    # H5
    nat = h5_natural(calibrator, thresholds[PRIMARY])
    hyp["H5"] = {**nat, "value": nat["difference"], "target": "difference >= 0", "pass": bool(nat["difference"] >= 0)}

    if hyp["H1"]["pass"] and hyp["H3"]["pass"]:
        overall = "SUCCESS"
    elif hyp["H1"]["pass"] or hyp["H4"]["pass"]:
        overall = "PARTIAL SUCCESS"
    else:
        overall = "FAILURE"

    secondary = {}
    for key, entry in held.items():
        if "|episode_counts" in key:
            continue
        secondary[key] = {m: {"t1_gradual": entry[m]["gradual"]["t1_share"], "t2_harmful": entry[m]["harmful"]["t2_share"],
                              "false_alerts": entry[m]["false_alerts_per_100_clean"],
                              "harmless_alert_share": entry[m]["harmless_alert_share"], "harm_auroc": entry[m]["harm_auroc"]}
                          for m in ev.METHODS}
    verdict = {"primary_setting": PRIMARY, "hypotheses": hyp, "overall": overall, "secondary": secondary,
               "day2_gate": "FAILED at every allowed shift strength (harmful share 0.17-0.19); proceeded at 1.5 by owner decision",
               "day4_gate": "PASSED (development T1 0.43 >= 0.40)"}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    write_markdown(verdict, ROOT / "docs" / "verdict.md")
    for h, v in hyp.items():
        print(h, "PASS" if v["pass"] else "FAIL", round(v["value"], 3) if isinstance(v["value"], float) else v["value"])
    print("OVERALL:", overall)


def f(v, p=2):
    return "n/a" if v is None else f"{v:.{p}f}"


def pretty(key):
    model, delay, labels = key.split("|")
    name = {"logistic_regression": "Logistic", "gradient_boosting": "Grad. boosting"}[model]
    return f"{name}, L = {delay[1:]}, {labels[6:]}% labels"


def write_markdown(v, path):
    h = v["hypotheses"]
    L = ["# v6 Verdict: ML Model Drift Early-Warning System", "",
         "Generated by `scripts/make_verdict.py` from the one-time held-out run. The protocol is in "
         "[v6_protocol.md](v6_protocol.md) and the plan in [final_two_week_plan.md](final_two_week_plan.md).", "",
         f"Primary setting: {pretty(v['primary_setting'])}. "
         "Held-out datasets: airlines, bank_marketing, electricity, ldpa, sea.", "",
         f"## Overall: **{v['overall']}**", "",
         "| ID | Hypothesis | Result (95% CI) | Target | Verdict |", "|---|---|---|---|---|",
         f"| H1 | Predictive warning (before the accuracy drop) for gradual harmful shifts | "
         f"{f(h['H1']['value'])} [{f(h['H1']['ci95'][0])}, {f(h['H1']['ci95'][1])}], median lead {f(h['H1']['median_lead'], 1)} batches, "
         f"{f(h['H1']['false_alerts'], 1)} false alerts / 100 clean batches (n = {h['H1']['episodes']}) | ≥ 0.60, lead ≥ 2, FA ≤ 5 | {'PASS' if h['H1']['pass'] else 'FAIL'} |",
         f"| H2 | Warns before labels can confirm, better than waiting for labels | "
         f"{f(h['H2']['t2_system'])} vs {f(h['H2']['t2_label_trend'])}, difference {f(h['H2']['value'])} "
         f"[{f(h['H2']['ci95'][0])}, {f(h['H2']['ci95'][1])}] (n = {h['H2']['episodes']}) | ≥ +0.10, CI > 0 | {'PASS' if h['H2']['pass'] else 'FAIL'} |",
         f"| H3 | Separates harmful from harmless shifts | AUROC {f(h['H3']['value'])} [{f(h['H3']['ci95'][0])}, {f(h['H3']['ci95'][1])}]; "
         f"harmless shifts with an alert {f(h['H3']['harmless_alert_share'])} [{f(h['H3']['harmless_alert_ci95'][0])}, {f(h['H3']['harmless_alert_ci95'][1])}] "
         f"({h['H3']['harmful']} harmful, {h['H3']['harmless']} harmless) | AUROC ≥ 0.80, harmless ≤ 0.10 | {'PASS' if h['H3']['pass'] else 'FAIL'} |",
         f"| H4 | Beats the original DEWS drift score | wins on {h['H4']['value']} of {h['H4']['datasets']} datasets | ≥ 4 of 5 | {'PASS' if h['H4']['pass'] else 'FAIL'} |",
         f"| H5 | Carries over to natural streams | {f(h['H5']['t2_system'])} vs {f(h['H5']['t2_label_trend'])}, difference {f(h['H5']['value'])} "
         f"[{f(h['H5']['ci95'][0])}, {f(h['H5']['ci95'][1])}] ({h['H5']['event_windows']} event windows, {h['H5']['datasets']} datasets) | ≥ 0 | {'PASS' if h['H5']['pass'] else 'FAIL'} |",
         "", "## Context the verdict does not capture", "",
         f"- **H4 compares unequal alert rates.** The original DEWS score's development threshold did not transfer: on held-out data it raised "
         f"{f(h['H4']['fixed_score_false_alerts'], 1)} false alerts per 100 clean batches, against a budget of 5. The primary system raised "
         f"{f(h['H4']['system_false_alerts'], 1)}. The DEWS score's higher warning rates come largely from alerting almost constantly. "
         "The rule was frozen without a budget condition, so the verdict stays FAIL.",
         f"- **H5 false alerts on natural streams:** system {f(h['H5']['false_alerts_per_100_clean']['primary_system'], 1)}, "
         f"label-trend {f(h['H5']['false_alerts_per_100_clean']['label_trend'], 1)} per 100 clean batches. "
         "Natural streams have no known shift onset, so T2 counts alerts from 5 batches before the event.",
         f"- **Gates:** Day 2 {v['day2_gate']}. Day 4 {v['day4_gate']}.",
         "", "### Per held-out dataset (H4)", "", "| Dataset | System T1 / T2 | DEWS score T1 / T2 | System wins |", "|---|---|---|---|"]
    for d, r in h["H4"]["per_dataset"].items():
        L.append(f"| {d} | {f(r['system_t1_t2'][0])} / {f(r['system_t1_t2'][1])} | {f(r['fixed_t1_t2'][0])} / {f(r['fixed_t1_t2'][1])} | {'yes' if r['system_wins'] else 'no'} |")
    L += ["", "## Secondary results (all settings, held-out; do not change the verdict)", "",
          "T1 = share of harmful gradual shifts warned before the drop; T2 = share of harmful shifts warned before labels confirm; FA = false alerts per 100 clean batches.", "",
          "| Setting | System T1 | System T2 | System FA | Label-trend T2 | Label-trend FA | DEWS score T1 | DEWS score FA |",
          "|---|---|---|---|---|---|---|---|"]
    for key, m in v["secondary"].items():
        s, l, d = m["primary_system"], m["label_trend"], m["fixed_dews_score"]
        L.append(f"| {pretty(key)} | {f(s['t1_gradual'])} | {f(s['t2_harmful'])} | {f(s['false_alerts'], 1)} | {f(l['t2_harmful'])} | {f(l['false_alerts'], 1)} | {f(d['t1_gradual'])} | {f(d['false_alerts'], 1)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
