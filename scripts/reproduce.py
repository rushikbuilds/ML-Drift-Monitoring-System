"""One command to reproduce the v6, v7 and v8 verdicts and the ablations.

    python scripts/reproduce.py --dry-run     # show the plan and what is already done
    python scripts/reproduce.py               # verify mode (default)
    python scripts/reproduce.py --refreeze    # also recompute development choices (overwrites frozen files)

Verify mode keeps the committed frozen thresholds and calibrators, rebuilds anything missing,
re-runs every confirmation, and checks that each verdict matches the committed one. Stages
whose outputs exist are skipped unless --force is given. Extraction runs as parallel shards.
Prerequisite for H5 only: the v2-v4 natural-stream caches (artifacts/v2/real_aux_signals.pkl,
new_signals.pkl), produced by the v2-v4 scripts listed in the README.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SHARDS = 6
A = ROOT / "artifacts"


def sharded(script: str, args: list[str]) -> list[list[str]]:
    return [[PY, script, *args, "--shard", str(i), "--shards", str(SHARDS)] for i in range(SHARDS)]


def stages(refreeze: bool):
    """(name, outputs that mark it done, commands run in parallel, needs --refreeze)."""
    s = [
        ("fetch datasets", [A / "v8" / "data_mv.pkl", A / "v7" / "data_adult.pkl"], [[PY, "scripts/fetch_datasets.py"]], False),
    ]
    for model in ("logistic_regression", "gradient_boosting"):
        s.append((f"v6 extract development ({model})", [A / "v6" / f"signals_development_{model}_5of6.pkl"],
                  sharded("scripts/run_v6_extract.py", ["--split", "development", "--model", model]), False))
    s += [("v6 develop (freeze thresholds)", [A / "v6" / "frozen_choices.json"], [[PY, "scripts/run_v6_develop.py"]], True)]
    for model in ("logistic_regression", "gradient_boosting"):
        s.append((f"v6 extract held-out ({model})", [A / "v6" / f"signals_held_out_{model}_5of6.pkl"],
                  sharded("scripts/run_v6_extract.py", ["--split", "held_out", "--model", model]), False))
    s += [
        ("v6 held-out run", [A / "v6" / "day5_held_out.json"], [[PY, "scripts/run_v6_heldout.py"]], False),
        ("v6 natural windows", [A / "v6" / "natural_signals_5of6.pkl"], sharded("scripts/run_v6_natural.py", []), False),
        ("v6 verdict", [], [[PY, "scripts/make_verdict.py"]], False),
        ("v7 develop (freeze thresholds)", [A / "v7" / "frozen_v7.json"], [[PY, "scripts/run_v7.py", "develop"]], True),
        ("v7 extract confirmation", [A / "v7" / "signals_confirmation_5of6.pkl"], sharded("scripts/run_v7.py", ["extract"]), False),
        ("v7 verdict", [], [[PY, "scripts/run_v7.py", "confirm"]], False),
        ("v8 harm signals (development)", [A / "v8" / "harm_development_5of6.pkl"],
         sharded("scripts/run_v8_extract.py", ["--split", "development"]), False),
        ("v8 develop (freeze threshold)", [A / "v8" / "frozen_v8.json"], [[PY, "scripts/run_v8.py", "develop"]], True),
        ("v8 extract confirmation", [A / "v8" / "signals_v8_confirmation_5of6.pkl"], sharded("scripts/run_v8.py", ["extract"]), False),
        ("v8 verdict", [], [[PY, "scripts/run_v8.py", "confirm"]], False),
        ("ablations", [], [[PY, "scripts/run_ablations.py"]], False),
    ]
    return s


VERDICTS = {"v6": A / "v6" / "verdict.json", "v7": A / "v7" / "v7_verdict.json", "v8": A / "v8" / "v8_verdict.json"}


def snapshot():
    out = {}
    for k, p in VERDICTS.items():
        if p.exists():
            v = json.loads(p.read_text())
            out[k] = {"overall": v["overall"], **{h: round(float(x["value"]), 6) for h, x in v["hypotheses"].items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run stages whose outputs already exist")
    ap.add_argument("--refreeze", action="store_true", help="recompute development choices (overwrites frozen files)")
    a = ap.parse_args()
    before = snapshot()
    for name, outputs, commands, freezes in stages(a.refreeze):
        done = bool(outputs) and all(p.exists() for p in outputs)
        if freezes and not a.refreeze:
            print(f"[keep]  {name}: using committed frozen file")
            continue
        if done and not a.force:
            print(f"[skip]  {name}: outputs present")
            continue
        print(f"[run]   {name} ({len(commands)} process{'es' if len(commands) > 1 else ''})", flush=True)
        if a.dry_run:
            continue
        procs = [subprocess.Popen(c, cwd=ROOT) for c in commands]
        if any(p.wait() != 0 for p in procs):
            raise SystemExit(f"stage failed: {name}")
    if a.dry_run:
        return
    after = snapshot()
    print("\nVerdict check against the state before this run:")
    ok = True
    for k in VERDICTS:
        same = before.get(k) == after.get(k)
        ok &= same or k not in before
        print(f"  {k}: {after.get(k, {}).get('overall')}  {'identical' if same else ('new' if k not in before else 'CHANGED')}")
    if not ok:
        raise SystemExit("a verdict changed; see the files under artifacts/ and docs/")


if __name__ == "__main__":
    main()
