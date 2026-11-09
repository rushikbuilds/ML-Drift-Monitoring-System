import altair as alt
import pandas as pd
import streamlit as st

from dashboard.app_data import (KIND_LABELS, SOURCES, STATUS, episode_facts, episode_label, load_episodes, replay,
                                signal_breakdown, visible_accuracy)
from dews import v7_config as v7cfg


@st.cache_data(max_entries=2, show_spinner="Loading episodes...")
def episodes(source):
    return load_episodes(source)


st.caption("Replays a saved episode batch by batch, exactly as the monitor saw it. "
           "Labels appear only after the selected delay. Thresholds are the frozen ones from development data.")

with st.container(border=True):
    row = st.columns([2, 2, 3, 2], vertical_alignment="bottom")
    source = row[0].selectbox("Data", list(SOURCES), key="source")
    eps = episodes(SOURCES[source])
    datasets = sorted({e["dataset"] for e in eps})
    dataset = row[1].selectbox("Dataset", datasets, key="dataset")
    kinds = [k for k in KIND_LABELS if any(e["dataset"] == dataset and e["kind"] == k for e in eps)]
    kind = row[2].segmented_control("Episode type", kinds, default="gradual_shift" if "gradual_shift" in kinds else kinds[0],
                                    format_func=KIND_LABELS.get, key="kind") or kinds[0]
    delay = row[3].segmented_control("Label delay (batches)", [0, 2, 5, 10], default=5, key="delay")
    delay = 5 if delay is None else delay
    choices = [e for e in eps if e["dataset"] == dataset and e["kind"] == kind]
    harmful_first = sorted(choices, key=lambda e: (e["category"] != "harmful", e["seed"]))
    picked = st.selectbox("Episode", range(len(harmful_first)), format_func=lambda i: episode_label(harmful_first[i]),
                          key="episode")
    ep = harmful_first[picked or 0]

frame = replay(ep, delay)
n = len(frame)
t = st.slider("Current batch", v7cfg.BURN_IN_BATCHES, n - 1, n - 1, key="batch",
              help="Move to see what the monitor knew at each point in time")
now = frame.iloc[t]
title, text = STATUS[now["status"]]
alert = {"calibrating": st.info, "healthy": st.success, "watch": st.info, "harm": st.warning, "confirmed": st.error}
alert[now["status"]](text, title=f"Batch {t}: {title}",
                     icon={"calibrating": ":material/hourglass_top:", "healthy": ":material/check_circle:",
                           "watch": ":material/visibility:", "harm": ":material/warning:",
                           "confirmed": ":material/error:"}[now["status"]])

def first_batch(mask):
    hits = frame["batch"][mask.fillna(False)]
    return int(hits.iloc[0]) if len(hits) else None


first = {"watch": first_batch(frame["watch_ratio"] >= 1), "harm": first_batch(frame["harm_ratio"] >= 1),
         "confirmed": first_batch(frame["status"] == "confirmed")}
facts = episode_facts(ep)
with st.container(horizontal=True):
    st.metric("Shift starts", facts["Shift starts at batch"] if facts["Shift starts at batch"] is not None else "none", border=True)
    st.metric("Accuracy event", facts["Accuracy event at batch"] if facts["Accuracy event at batch"] is not None else "none", border=True)
    st.metric("First drift watch", first["watch"] if first["watch"] is not None else "none", border=True)
    st.metric("First harm warning", first["harm"] if first["harm"] is not None else "none", border=True)
    st.metric("Labels confirm drop", first["confirmed"] if first["confirmed"] is not None else "none", border=True)

marks = []
if facts["Shift starts at batch"] is not None:
    marks.append({"batch": facts["Shift starts at batch"], "what": "Shift starts"})
if facts["Accuracy event at batch"] is not None:
    marks.append({"batch": facts["Accuracy event at batch"], "what": "Accuracy event"})
marks.append({"batch": t, "what": "Current batch"})
marks = pd.DataFrame(marks)
marker_base = alt.Chart(marks).encode(x="batch:Q", tooltip=["what", "batch"])
rules = (marker_base.mark_rule(color="gray", strokeDash=[4, 4])
         + marker_base.mark_text(align="left", baseline="top", dx=3, y=4, color="gray", fontSize=11).encode(text="what:N"))
TIER_COLORS = alt.Scale(domain=["Drift watch", "Harm warning"], range=["#4c9be8", "#f28e2b"])

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Monitor scores", anchor=False)
        st.caption("Each score divided by its alert threshold: above 1 means an alert.")
        scores = frame[frame["batch"] <= t].melt("batch", ["watch_ratio", "harm_ratio"], "tier", "score")
        scores["tier"] = scores["tier"].map({"watch_ratio": "Drift watch", "harm_ratio": "Harm warning"})
        lines = alt.Chart(scores.dropna()).mark_line(point=True).encode(
            x=alt.X("batch:Q", scale=alt.Scale(domain=[0, n - 1]), title="Batch"),
            y=alt.Y("score:Q", title="Score / threshold"),
            color=alt.Color("tier:N", title=None, scale=TIER_COLORS, legend=alt.Legend(orient="bottom")),
            tooltip=["batch", "tier", alt.Tooltip("score:Q", format=".2f")])
        threshold = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="gray").encode(y="y:Q")
        st.altair_chart(lines + threshold + rules, alt="Drift watch and harm warning scores relative to their thresholds")
with right:
    with st.container(border=True):
        st.subheader("Accuracy", anchor=False)
        st.caption("Green: what delayed labels have shown so far. Grey: the true accuracy, hidden from the monitor. "
                   "Red dotted line: 5 points below the starting accuracy, where an accuracy event begins.")
        acc = pd.DataFrame({"batch": frame["batch"], "Seen through delayed labels": visible_accuracy(frame, t),
                            "True accuracy (hidden)": frame["true_accuracy"],
                            "Harm estimate (no labels)": frame["estimated_accuracy"].where(frame["batch"] <= t)})
        acc = acc.melt("batch", var_name="series", value_name="accuracy").dropna()
        accuracy = alt.Chart(acc).mark_line().encode(
            x=alt.X("batch:Q", scale=alt.Scale(domain=[0, n - 1]), title="Batch"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False), title="Accuracy"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom", columns=1),
                            scale=alt.Scale(domain=["Seen through delayed labels", "Harm estimate (no labels)", "True accuracy (hidden)"],
                                            range=["#59a14f", "#f28e2b", "#bab0ac"])),
            opacity=alt.condition(alt.datum.series == "True accuracy (hidden)", alt.value(0.3), alt.value(1.0)),
            tooltip=["batch", "series", alt.Tooltip("accuracy:Q", format=".3f")])
        drop = alt.Chart(pd.DataFrame({"y": [frame["baseline_accuracy"].iloc[0] - 0.05]})).mark_rule(
            color="firebrick", strokeDash=[2, 2]).encode(y="y:Q")
        st.altair_chart(accuracy + drop + rules, alt="Observed, true and estimated accuracy over time")

with st.container(border=True):
    st.subheader(f"Why the drift watch score is {now['watch_score']:.1f} at batch {t}", anchor=False)
    st.caption(f"Each signal in standard deviations from this stream's first {v7cfg.BURN_IN_BATCHES} batches. "
               "The drift watch score is the largest of them.")
    breakdown = signal_breakdown(ep, delay, t)
    st.altair_chart(alt.Chart(breakdown).mark_bar().encode(
        x=alt.X("z:Q", title="Standard deviations from normal"), y=alt.Y("signal:N", sort="-x", title=None),
        tooltip=["signal", alt.Tooltip("z:Q", format=".1f")]), alt="Contribution of each drift signal")

with st.expander("Episode details"):
    st.table(pd.DataFrame({"Value": {k: ("—" if v is None else str(v)) for k, v in facts.items()}}))
