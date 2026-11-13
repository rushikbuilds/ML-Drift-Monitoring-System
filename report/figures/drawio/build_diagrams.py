"""Write the report's draw.io diagrams (*.drawio, editable in draw.io / diagrams.net).

    python report/figures/drawio/build_diagrams.py      # writes the .drawio files
    python report/figures/drawio/export_drawio.py       # exports them to report/figures/*.pdf
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
FONT = "fontFamily=Helvetica;"

# professional palette: fill / stroke
BLUE = ("#EFF6FF", "#2563EB")
PURPLE = ("#F5F3FF", "#7C3AED")
GREEN = ("#ECFDF5", "#059669")
ORANGE = ("#FFF7ED", "#EA580C")
RED = ("#FEF2F2", "#DC2626")
GRAY = ("#F8FAFC", "#64748B")
YELLOW = ("#FFFBEB", "#D97706")
NAVY = ("#1E3A8A", "#1E3A8A")
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#475569;strokeWidth=1.5;fontColor=#334155;fontSize=12;" + FONT


class Diagram:
    def __init__(self, name: str):
        self.name, self.cells, self.n = name, [], 0

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, text, color=BLUE, shape="rounded=1;arcSize=14;", size=13, parent="1", extra=""):
        i = self._id()
        fc, sc = color
        font = "#FFFFFF" if color == NAVY else "#0F172A"
        if color == NAVY:
            text = text.replace("#475569", "#CBD5E1")
        style = (f"{shape}whiteSpace=wrap;html=1;fillColor={fc};strokeColor={sc};fontColor={font};"
                 f"fontSize={size};strokeWidth=1.5;{FONT}{extra}")
        self.cells.append(f'<mxCell id="{i}" value="{escape(text, {chr(34): "&quot;"})}" style="{style}" vertex="1" '
                          f'parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, x, y, w, h, text, size=13, extra="", parent="1"):
        i = self._id()
        style = f"text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fontSize={size};fontColor=#0F172A;{FONT}{extra}"
        self.cells.append(f'<mxCell id="{i}" value="{escape(text, {chr(34): "&quot;"})}" style="{style}" vertex="1" '
                          f'parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def group(self, x, y, w, h, title, color=("#F8FAFC", "#CBD5E1"), size=14, horizontal=True):
        i = self._id()
        hz = "" if horizontal else "horizontal=0;"
        style = (f"swimlane;{hz}startSize=30;html=1;rounded=1;arcSize=3;fillColor={color[0]};swimlaneFillColor={color[0]};"
                 f"strokeColor={color[1]};fontStyle=1;fontSize={size};fontColor=#0F172A;{FONT}")
        self.cells.append(f'<mxCell id="{i}" value="{escape(title)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, s, t, label="", extra="", parent="1"):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{EDGE}{extra}" edge="1" parent="{parent}" '
                          f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')
        return i

    def write(self):
        xml = ('<mxGraphModel adaptiveColors="none" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
               + "".join(self.cells) + "</root></mxGraphModel>")
        (HERE / f"{self.name}.drawio").write_text(xml, encoding="utf-8")


def small(t):
    return f"<font style='font-size:11px;color:#475569'>{t}</font>"


def b(t):
    return f"<b>{t}</b>"


# 1. Overall system architecture -------------------------------------------------------------
def architecture():
    d = Diagram("fig_architecture")
    W, H, LH = 210, 70, 112
    names = ["Data", "Model", "Signals", "Monitor", "Outputs"]
    L = [d.group(20, 20 + k * (LH + 8), 1090, LH, n, horizontal=False) for k, n in enumerate(names)]
    x = lambda c: 55 + c * 260  # noqa: E731
    y = 21
    ds = d.box(x(0), y, W, H, b("OpenML datasets") + "<br>" + small("24 public tabular sets"), PURPLE,
               shape="shape=cylinder3;boundedLbl=1;size=10;", parent=L[0])
    gen = d.box(x(1), y, W, H, b("Benchmark generator") + "<br>" + small("re-sampled episodes, known onset;<br>or natural streams"), BLUE, parent=L[0])
    batch = d.box(x(1), y, W, H, b("Unlabeled batch B<sub>t</sub>") + "<br>" + small("300 rows, labels hidden"), BLUE, parent=L[1])
    model = d.box(x(2), y, W, H, b("Frozen classifier f") + "<br>" + small("logistic regression"), NAVY, parent=L[1])
    ref = d.box(x(3), y, W, H, b("Labeled reference R") + "<br>" + small("2000 rows: fit + importance"), PURPLE,
                shape="shape=cylinder3;boundedLbl=1;size=10;", parent=L[1])
    inp = d.box(x(1), y, W, H, b("Input-shift signals") + "<br>" + small("KS, PSI, MMD, DEWS score,<br>relevance-weighted shift"), BLUE, parent=L[2])
    beh = d.box(x(2), y, W, H, b("Model-behavior signals") + "<br>" + small("confidence, entropy,<br>auxiliary-model disagreement"), BLUE, parent=L[2])
    watch = d.box(x(1.5), y, W, H, b("Drift watch (v7)") + "<br>" + small("robust z vs burn-in, max &ge; 4.0"), GREEN, parent=L[3])
    harm = d.box(x(3), y, W, H, b("Harm tier (v8)") + "<br>" + small("importance-weighted accuracy<br>of f on R (vs B<sub>t</sub>), drop &ge; 1.15 pts"), ORANGE, parent=L[3])
    lab = d.box(x(3), y, W, H, b("Delayed labels") + "<br>" + small("usable after L batches"), ORANGE, parent=L[4])
    alert = d.box(x(2), y, W, H, b("Alerts") + "<br>" + small("WATCH and HARM per batch"), RED, parent=L[4])
    ver = d.box(x(1), y, W, H, b("Evaluation") + "<br>" + small("tiers T1/T2, verdict scripts"), BLUE, parent=L[4])
    dash = d.box(x(0), y, W, H, b("Dashboard and reports") + "<br>" + small("Streamlit, paper, report"), GRAY, parent=L[4])
    d.edge(ds, gen, "rows")
    d.edge(gen, batch, "stream")
    d.edge(gen, ref, "reference block", extra="exitX=1;exitY=0.5;entryX=0.5;entryY=0;")
    d.edge(batch, model, "predict")
    d.edge(ref, model, "fit")
    d.edge(model, inp, extra="exitX=0.15;exitY=1;entryX=0.85;entryY=0;"); d.edge(model, beh)
    d.edge(inp, watch); d.edge(beh, watch)
    d.edge(ref, harm, "rows + correctness")
    d.edge(watch, alert); d.edge(harm, alert)
    d.edge(lab, alert, "confirm / clear", extra="dashed=1;")
    d.edge(alert, ver); d.edge(ver, dash)
    d.write()


# 2. Two-tier monitor, per batch ---------------------------------------------------------------
def two_tier():
    d = Diagram("fig_two_tier_flow")
    lanes = [("Drift watch tier (v7)", GREEN), ("Harm tier (v8)", ORANGE), ("Delayed labels", GRAY)]
    LH, W = 150, 1120
    L = [d.group(20, 20 + k * (LH + 10), W, LH, n, (c[0], c[1]), horizontal=False) for k, (n, c) in enumerate(lanes)]
    bw, bh, y = 150, 70, 40
    xs = [50 + k * 175 for k in range(6)]
    s1 = d.box(xs[0], y, bw, bh, b("Batch B<sub>t</sub>") + "<br>" + small("frozen f predicts"), BLUE, parent=L[0])
    s2 = d.box(xs[1], y, bw, bh, b("7 signals") + "<br>" + small("shift + behavior"), BLUE, parent=L[0])
    s3 = d.box(xs[2], y, bw, bh, b("Robust z-score") + "<br>" + small("median, MAD of batches 0-7"), BLUE, parent=L[0])
    s4 = d.box(xs[3], y, bw, bh, b("Max over signals") + "<br>" + small("watch score"), BLUE, parent=L[0])
    s5 = d.box(xs[4] - 5, y - 10, 160, 90, "score &ge; 4.0?", YELLOW, shape="rhombus;", size=13, parent=L[0])
    s6 = d.box(xs[5] + 5, y, bw, bh, b("WATCH alert") + "<br>" + small("drift under way"), GREEN, parent=L[0])
    h1 = d.box(xs[0], y, bw, bh, b("Domain classifier g<sub>t</sub>") + "<br>" + small("reference vs batch"), BLUE, parent=L[1])
    h2 = d.box(xs[1], y, bw, bh, b("Importance weights") + "<br>" + small("odds, capped at 20"), BLUE, parent=L[1])
    h3 = d.box(xs[2], y, bw, bh, b("Weighted accuracy") + "<br>" + small("of f on reference rows"), BLUE, parent=L[1])
    h4 = d.box(xs[3], y, bw, bh, b("Harm score h<sub>t</sub>") + "<br>" + small("burn-in median - estimate"), BLUE, parent=L[1])
    h5 = d.box(xs[4] - 5, y - 10, 160, 90, "h<sub>t</sub> &ge; 1.15 pts?", YELLOW, shape="rhombus;", parent=L[1])
    h6 = d.box(xs[5] + 5, y, bw, bh, b("HARM alert") + "<br>" + small("accuracy likely falling"), ORANGE, parent=L[1])
    c1 = d.box(xs[0], y, bw, bh, b("Labels of B<sub>t-L</sub>") + "<br>" + small("arrive after delay L"), ORANGE, parent=L[2])
    c2 = d.box(xs[1], y, bw, bh, b("Observed accuracy") + "<br>" + small("vs mean of batches 0-4"), BLUE, parent=L[2])
    c3 = d.box(xs[2] - 5, y - 10, 160, 90, "2 batches &ge; 5 pts low?", YELLOW, shape="rhombus;", parent=L[2])
    c4 = d.box(xs[3] + 50, y, bw, bh, b("Event confirmed") + "<br>" + small("warning was true"), RED, parent=L[2])
    c5 = d.box(xs[5] + 5, y, bw, bh, b("Alert cleared") + "<br>" + small("harmless or no shift"), GREEN, parent=L[2])
    for a, z in [(s1, s2), (s2, s3), (s3, s4), (s4, s5), (h1, h2), (h2, h3), (h3, h4), (h4, h5), (c1, c2), (c2, c3)]:
        d.edge(a, z)
    d.edge(s5, s6, "yes", extra="fontColor=#059669;")
    d.edge(h5, h6, "yes", extra="fontColor=#059669;")
    d.edge(c3, c4, "yes", extra="fontColor=#059669;")
    d.edge(c3, c5, "no", extra="fontColor=#DC2626;exitX=0.5;exitY=1;entryX=0.5;entryY=1;")
    d.edge(s1, h1, "same batch", extra="dashed=1;")
    d.write()


# 3. Project evolution v1-v10 ------------------------------------------------------------------
def evolution():
    d = Diagram("fig_evolution")
    items = [
        ("v1", "Risk calibrator on synthetic data", "0 of 20 events detected", RED),
        ("v2", "Redesigned scenarios, state target", "Synthetic gain; real data negative", ORANGE),
        ("v3", "Delay-aware CBPE (accuracy units)", "Transfers; little early warning", ORANGE),
        ("v4", "Delay-anchored accuracy estimator", "Lower nowcast error; trade-off", ORANGE),
        ("v5", "Harmless vs harmful triage", "Works on synthetic, not on real", RED),
        ("v6", "Pre-registered real-data benchmark", "FAILURE: H1-H5 all fail", RED),
        ("v7", "Stream-normalized drift watch", "PARTIAL SUCCESS: T1 0.87", GREEN),
        ("v8", "Importance-weighted harm tier", "PARTIAL SUCCESS: AUROC 0.78", GREEN),
        ("v9", "Nonlinear harm estimators", "Not adopted: dev gate failed", GRAY),
        ("v10", "Stricter alerting rules", "Not adopted: T2 0.30 on fresh data", GRAY),
    ]
    badges = []
    d.group(15, 45, 1110, 200, "Exploratory phase (v1-v5)")
    d.group(15, 265, 1110, 200, "Pre-registered phase (v6-v10)")
    for k, (v, what, res, col) in enumerate(items):
        row, c = divmod(k, 5)
        x = 30 + c * 220
        y = 85 + row * 220
        bdg = d.box(x + 50, y, 100, 36, b(v), NAVY, extra="arcSize=50;")
        d.box(x, y + 50, 200, 100, b(what) + "<br><br>" + small(res), col)
        badges.append(bdg)
    for k in range(9):
        extra = "exitX=0.5;exitY=1;entryX=0.5;entryY=0;" if k == 4 else ""
        if k != 4:
            d.edge(badges[k], badges[k + 1], "", extra=extra)
    d.text(30, 475, 1080, 24, "<span style='color:#DC2626'>&#9632;</span> failed&nbsp;&nbsp;&nbsp;"
           "<span style='color:#EA580C'>&#9632;</span> mixed / exploratory&nbsp;&nbsp;&nbsp;"
           "<span style='color:#059669'>&#9632;</span> adopted (partial success)&nbsp;&nbsp;&nbsp;"
           "<span style='color:#64748B'>&#9632;</span> pre-registered follow-up, not adopted&nbsp;&nbsp;&nbsp;"
           "", size=12)
    d.write()


# 4. Pre-registration workflow -----------------------------------------------------------------
def preregistration():
    d = Diagram("fig_preregistration")
    W, H = 210, 64
    X = [40, 330, 620]
    Y = lambda r: 20 + r * 105  # noqa: E731
    s = d.box(X[0], Y(0), W, 50, b("New idea after a verdict"), NAVY, extra="arcSize=50;")
    p1 = d.box(X[0], Y(1), W, H, b("Write protocol") + "<br>" + small("hypotheses, pass rules, datasets"), BLUE)
    p2 = d.box(X[0], Y(2), W, H, b("Commit config module") + "<br>" + small("timestamped in git"), BLUE)
    p3 = d.box(X[0], Y(3), W, H, b("Develop on development data") + "<br>" + small("choose thresholds and rules"), BLUE)
    g1 = d.box(X[0] + 15, Y(4) - 10, 180, 90, "Development gate passed?", YELLOW, shape="rhombus;")
    stop = d.box(X[0], Y(5) + 15, W, H, b("Stop, report negative") + "<br>" + small("no fresh data used (v9)"), RED)
    p4 = d.box(X[1], Y(0), W, H, b("Freeze thresholds") + "<br>" + small("SHA-256 fingerprint in protocol"), BLUE)
    p5 = d.box(X[1], Y(1), W, H, b("Download confirmation data") + "<br>" + small("datasets never used before"), PURPLE)
    g2 = d.box(X[1] + 15, Y(2) - 10, 180, 90, "Fingerprint matches?", YELLOW, shape="rhombus;")
    ref = d.box(X[2], Y(2), W, H, b("Refuse to run"), RED)
    p6 = d.box(X[1], Y(3), W, H, b("One-time confirmation run"), BLUE)
    p7 = d.box(X[1], Y(4), W, H, b("Verdict script") + "<br>" + small("applies the frozen pass rules"), BLUE)
    out = d.box(X[1], Y(5) + 15, W, H, b("Publish verdict") + "<br>" + small("success, partial success or failure"), GREEN)
    d.box(X[2], Y(4) - 10, W, 100, b("Deviation log") + "<br>" + small("every change dated, with its reason,<br>recorded before any affected<br>episode is built"), YELLOW,
          shape="shape=note;size=16;")
    for a, z in [(s, p1), (p1, p2), (p2, p3), (p3, g1), (p4, p5), (p5, g2), (p6, p7), (p7, out)]:
        d.edge(a, z)
    d.edge(g1, p4, "yes", extra="fontColor=#059669;exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.edge(g1, stop, "no", extra="fontColor=#DC2626;")
    d.edge(g2, ref, "no", extra="fontColor=#DC2626;")
    d.edge(g2, p6, "yes", extra="fontColor=#059669;")
    d.write()


# 5. Benchmark episode generation --------------------------------------------------------------
def episode():
    d = Diagram("fig_episode_generation")
    W, H = 220, 62
    X = [30, 310, 590]
    Y = lambda r: 20 + r * 92  # noqa: E731
    st = d.box(X[0], Y(0), W, 46, b("Dataset, episode kind, seed"), NAVY, extra="arcSize=50;")
    p1 = d.box(X[0], Y(1), W, H, b("Draw contiguous block") + "<br>" + small("both classes &ge; 5% of rows"), BLUE)
    p2 = d.box(X[0], Y(2), W, H, b("Reference: 2000 rows") + "<br>" + small("fit f on 1500, importance on 500"), PURPLE)
    p3 = d.box(X[0], Y(3), W, H, b("Pool: next 20,000 rows") + "<br>" + small("same period, no built-in drift"), PURPLE)
    p4 = d.box(X[0], Y(4), W, H, b("Draw onset s in [10, 16]"), BLUE)
    k = d.box(X[0] + 20, Y(5) - 14, 180, 90, "Episode kind?", YELLOW, shape="rhombus;")
    cl = d.box(X[1], Y(6) + 20, W, H, b("Clean") + "<br>" + small("uniform sampling"), GREEN)
    sh = d.box(X[1], Y(5), W, 76, b("Gradual / abrupt shift") + "<br>" + small("feature j from important or<br>unimportant half; weights &prop; exp(&lambda;(t) &sigma; z<sub>j</sub>)"), ORANGE)
    lc = d.box(X[1], Y(7) + 40, W, H, b("Label change") + "<br>" + small("flip labels, p &rarr; 0.3, if z<sub>k</sub> &gt; 0"), RED)
    smp = d.box(X[2], Y(5), W, H, b("Sample 45 batches") + "<br>" + small("300 rows each, with replacement"), BLUE)
    acc = d.box(X[2], Y(6) + 20, W, H, b("True accuracy of f") + "<br>" + small("per batch, from the kept labels"), BLUE)
    ev = d.box(X[2] + 20, Y(7) + 30, 180, 90, "Event after onset?", YELLOW, shape="rhombus;")
    hf = d.box(X[2], Y(9) + 10, W, H, b("Harmful episode") + "<br>" + small("event at or after onset"), RED)
    hl = d.box(X[1], Y(8) + 44, W, H, b("Harmless episode") + "<br>" + small("no event"), GREEN)
    for a, z in [(st, p1), (p1, p2), (p2, p3), (p3, p4), (p4, k), (sh, smp), (smp, acc), (acc, ev)]:
        d.edge(a, z)
    d.edge(k, cl, "clean", extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5;")
    d.edge(k, sh, "shift")
    d.edge(k, lc, "label change", extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    d.edge(cl, smp, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    d.edge(lc, smp, extra="exitX=1;exitY=0.5;entryX=0;entryY=1;")
    d.edge(ev, hf, "yes", extra="fontColor=#DC2626;")
    d.edge(ev, hl, "no", extra="fontColor=#059669;exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.text(X[0], Y(8) + 20, W + 10, 90, small("Spurious episodes (event before onset) are excluded.<br>Per dataset: 10 clean, 30 gradual,<br>15 abrupt and 10 label-change episodes."), size=11, extra="align=left;")
    d.write()


# 6. Warning tiers on a time axis ---------------------------------------------------------------
def tiers():
    d = Diagram("fig_warning_tiers")
    X0, U = 60, 22  # x of batch 0, px per batch
    xb = lambda t: X0 + t * U  # noqa: E731
    s, e, L = 12, 20, 5
    conf = e + 1 + max(L, 1)
    d.box(xb(0), 120, xb(45) - xb(0), 4, "", ("#94A3B8", "#94A3B8"), shape="")
    d.box(xb(0), 90, xb(8) - xb(0), 26, small("burn-in, no alerts"), GRAY, shape="")
    d.box(xb(0), 60, xb(5) - xb(0), 26, small("baseline 0-4"), PURPLE, shape="")
    for t in range(0, 46, 5):
        d.text(xb(t) - 15, 128, 30, 18, small(str(t)), size=10)
    d.text(xb(45) + 10, 113, 70, 18, small("batch"), size=10)
    for t, name, col in [(s, "intervention s", "#2563EB"), (e, "accuracy event e", "#DC2626"), (conf, "labels confirm e+1+L", "#EA580C")]:
        d.box(xb(t) - 1, 40, 2, 120, "", (col, col), shape="")
        d.text(xb(t) - 70, 15, 140, 22, f"<b style='color:{col}'>{name}</b>", size=12)
    d.box(xb(s), 175, xb(e) - xb(s), 34, b("T1 predictive"), GREEN, shape="")
    d.box(xb(s), 215, xb(conf) - xb(s), 34, b("T2 pre-confirmation"), ORANGE, shape="")
    d.box(xb(conf), 255, xb(45) - xb(conf), 34, b("T3 confirmed (labels already show it)"), GRAY, shape="")
    d.text(xb(0), 300, 1000, 22, small("Example with onset s = 12, event e = 20, label delay L = 5. Lead time = e minus the first alert in the T1 window."), size=11, extra="align=left;")
    d.write()


# 7. Software modules ------------------------------------------------------------------------------
def modules():
    d = Diagram("fig_modules")
    W, H = 230, 120
    core = d.box(330, 170, 260, 150, b("dews/ (core library)") + "<br>" + small("resample_benchmark, v6_signals,<br>v6_eval, v7_monitor,<br>harm_estimation, detection,<br>v2, anchored"), NAVY)
    scr = d.box(30, 170, W, 150, b("scripts/") + "<br>" + small("run_v6_*, run_v7, run_v8,<br>make_verdict, run_ablations,<br>run_posthoc, reproduce,<br>fetch_datasets"), BLUE)
    cfg = d.box(330, 20, 260, 90, b("Frozen configs") + "<br>" + small("v6/v7/v8/v10_config.py,<br>frozen_*.json + SHA-256"), YELLOW)
    art = d.box(30, 400, W, H, b("artifacts/") + "<br>" + small("episodes (.pkl), verdicts,<br>ablations, post hoc JSON"), PURPLE,
                shape="shape=cylinder3;boundedLbl=1;size=12;")
    app = d.box(330, 400, 260, H, b("Streamlit dashboard") + "<br>" + small("streamlit_app.py, app_pages/,<br>dashboard/app_data.py"), GREEN)
    tst = d.box(660, 170, W, 150, b("tests/") + "<br>" + small("85 unit tests: protocol,<br>label maturity, signals,<br>monitor, dashboard data"), BLUE)
    ci = d.box(660, 20, W, 90, b("GitHub Actions CI") + "<br>" + small("Ruff + pytest on 3.10-3.12,<br>benchmark gate"), GRAY)
    doc = d.box(660, 400, W, H, b("docs/ and paper/") + "<br>" + small("protocols, verdicts,<br>IEEE paper, this report"), GRAY)
    d.edge(scr, core, "uses"); d.edge(cfg, core, "read by"); d.edge(scr, art, "writes")
    d.edge(art, app, "reads"); d.edge(tst, core, "tests"); d.edge(ci, tst, "runs")
    d.edge(art, doc, "figures and numbers", extra="exitX=0.5;exitY=1;entryX=0.5;entryY=1;")
    d.write()


# 8. Use cases ------------------------------------------------------------------------------------
def usecases():
    d = Diagram("fig_usecases")
    sysb = d.group(200, 20, 470, 470, "Drift Early-Warning System", ("#F8FAFC", "#94A3B8"))
    uc = []
    names = ["Replay an episode batch by batch", "See drift-watch and harm alerts", "Inspect which signals moved",
             "Read verdicts and ablations", "Reproduce frozen verdicts", "Run a pre-registered experiment"]
    for k, n in enumerate(names):
        uc.append(d.box(105, 45 + k * 70, 260, 52, n, BLUE, shape="ellipse;", parent=sysb))
    eng = d.box(60, 140, 40, 70, "ML engineer", ("#FFFFFF", "#0F172A"),
                shape="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;outlineConnect=0;")
    res = d.box(60, 360, 40, 70, "Researcher / examiner", ("#FFFFFF", "#0F172A"),
                shape="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;outlineConnect=0;")
    oml = d.box(760, 400, 40, 70, "OpenML", ("#FFFFFF", "#0F172A"),
                shape="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;outlineConnect=0;")
    line = "endArrow=none;edgeStyle=none;"
    for k in (0, 1, 2):
        d.edge(eng, uc[k], extra=line)
    for k in (3, 4, 5):
        d.edge(res, uc[k], extra=line)
    d.edge(oml, uc[5], extra=line); d.edge(oml, uc[4], extra=line)
    d.write()


# 9. Dashboard data flow ---------------------------------------------------------------------------
def dashboard():
    d = Diagram("fig_dashboard_flow")
    W, H = 200, 70
    a1 = d.box(30, 30, W, H, b("Episode pickles") + "<br>" + small("v7 / v8 confirmation"), PURPLE, shape="shape=cylinder3;boundedLbl=1;size=10;")
    a2 = d.box(30, 140, W, H, b("Verdict JSON") + "<br>" + small("v6, v7, v8"), PURPLE, shape="shape=cylinder3;boundedLbl=1;size=10;")
    a3 = d.box(30, 250, W, H, b("Ablations JSON"), PURPLE, shape="shape=cylinder3;boundedLbl=1;size=10;")
    core = d.box(300, 120, 220, 110, b("dashboard/app_data.py") + "<br>" + small("load, replay with label delay,<br>tier scores / thresholds,<br>signal breakdown"), NAVY)
    pages = [("Overview", "headline results"), ("Live monitor", "batch slider, both tiers"), ("Verdicts", "all hypotheses"), ("Ablations", "why it works or fails")]
    ps = [d.box(600, 20 + k * 85, W, 66, b(n) + "<br>" + small(t), GREEN) for k, (n, t) in enumerate(pages)]
    user = d.box(870, 140, 40, 70, "User", ("#FFFFFF", "#0F172A"), shape="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;outlineConnect=0;")
    for a in (a1, a2, a3):
        d.edge(a, core)
    for p in ps:
        d.edge(core, p, extra="edgeStyle=none;")
        d.edge(p, user, extra="endArrow=none;edgeStyle=none;")
    d.write()


# 10. CI pipeline ------------------------------------------------------------------------------------
def ci():
    d = Diagram("fig_ci")
    W, H, y = 170, 64, 60
    steps = [(b("Push / pull request"), NAVY), (b("Install") + "<br>" + small("requirements.txt"), BLUE),
             (b("Ruff lint") + "<br>" + small("pinned 0.17.4"), BLUE), (b("pytest") + "<br>" + small("84 pass, 1 skip"), BLUE),
             (b("Benchmark gate") + "<br>" + small("lag &le; 3, alerts &ge; 1"), GREEN)]
    g = d.group(230, 20, 3 * (W + 30) + 10, 150, "Matrix: Python 3.10, 3.11, 3.12", ("#F8FAFC", "#CBD5E1"))
    ids = [d.box(30, y, W, H, steps[0][0], steps[0][1])]
    for k in range(1, 4):
        ids.append(d.box(15 + (k - 1) * (W + 30), 55, W, H, steps[k][0], steps[k][1], parent=g))
    ids.append(d.box(230 + 3 * (W + 30) + 50, y, W, H, steps[4][0], steps[4][1]))
    for k in range(4):
        d.edge(ids[k], ids[k + 1])
    d.write()


# Paper (single IEEE column) ------------------------------------------------------------------------
def paper_system():
    d = Diagram("paper_system")
    W, H = 118, 52
    bt = d.box(10, 10, W, H, b("Batch B<sub>t</sub>") + "<br>" + small("no labels"), BLUE, size=12)
    f = d.box(150, 10, W, H, b("Frozen model f"), NAVY, size=12)
    r = d.box(290, 6, W, 60, b("Labeled<br>reference R"), PURPLE, shape="shape=cylinder3;boundedLbl=1;size=8;", size=12)
    sg = d.box(10, 100, 258, H, b("7 label-free signals") + "<br>" + small("input shift, confidence, disagreement"), BLUE, size=12)
    w = d.box(10, 190, W + 20, H + 6, b("Drift watch") + "<br>" + small("max robust z vs burn-in"), GREEN, size=12)
    h = d.box(230, 190, 178, H + 6, b("Harm tier") + "<br>" + small("importance-weighted accuracy<br>of f on R, reweighted to B<sub>t</sub>"), ORANGE, size=12)
    a = d.box(140, 290, W, H, b("Alerts"), RED, size=12)
    lb = d.box(330, 290, W, H, b("Delayed labels") + "<br>" + small("after L batches"), ORANGE, size=12)
    d.edge(bt, f); d.edge(r, f, "fit")
    d.edge(f, sg); d.edge(sg, w)
    d.edge(r, h, "rows")
    d.edge(w, a); d.edge(h, a)
    d.edge(lb, a, "confirm / clear", extra="dashed=1;fontSize=11;")
    d.write()


def paper_tiers():
    d = Diagram("paper_tiers")
    X0, U = 30, 11
    xb = lambda t: X0 + t * U  # noqa: E731
    s, e, L = 12, 20, 5
    conf = e + 1 + max(L, 1)
    d.box(xb(0), 70, xb(40) - xb(0), 3, "", ("#94A3B8", "#94A3B8"), shape="")
    d.box(xb(0), 46, xb(8) - xb(0), 20, small("burn-in"), GRAY, shape="", size=11)
    for t in range(0, 41, 10):
        d.text(xb(t) - 12, 75, 24, 16, small(str(t)), size=10)
    for t, name, col, al in [(s, "s: shift", "#2563EB", "r"), (e, "e: accuracy drop", "#DC2626", "l"), (conf, "labels confirm", "#EA580C", "l2")]:
        d.box(xb(t) - 1, 22, 2, 70, "", (col, col), shape="")
        x = {"r": xb(t) - 112, "l": xb(t) + 3, "l2": xb(t) + 3}[al]
        d.text(x, 22 if al == "l2" else 2, 110, 18, f"<b style='color:{col}'>{name}</b>", size=11,
               extra={"r": "align=right;", "l": "align=left;", "l2": "align=left;"}[al])
    d.box(xb(s), 98, xb(e) - xb(s), 22, b("T1"), GREEN, shape="", size=11)
    d.box(xb(s), 124, xb(conf) - xb(s), 22, b("T2"), ORANGE, shape="", size=11)
    d.box(xb(conf), 150, xb(40) - xb(conf), 22, b("T3"), GRAY, shape="", size=11)
    d.write()


if __name__ == "__main__":
    for fn in (architecture, two_tier, evolution, preregistration, episode, tiers, modules, usecases, dashboard, ci, paper_system, paper_tiers):
        fn()
    print("written:", sorted(p.name for p in HERE.glob("*.drawio")))
