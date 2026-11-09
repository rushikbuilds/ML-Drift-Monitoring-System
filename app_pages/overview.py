import pandas as pd
import streamlit as st

from dashboard.app_data import load_results


@st.cache_data(max_entries=1)
def results():
    return load_results()


r = results()
v7, v8 = r["v7"], r["v8"]
c1 = v7["hypotheses"]["C1"]
d3, d4 = v8["hypotheses"]["D3"], v8["hypotheses"]["D4"]

st.markdown(
    "A monitor for deployed classifiers whose **labels arrive late**. It has two warning tiers. "
    "**Drift watch** says the inputs or the model's behaviour have drifted. "
    "**Harm warning** says the drift is moving data into regions where the model makes more mistakes. "
    "Delayed labels then **confirm** or clear the warning. "
    "All numbers below come from one-time runs on datasets the system had never seen."
)

with st.container(horizontal=True):
    st.metric("Harmful drifts warned before accuracy drops", f"{c1['value']:.0%}", border=True,
              help="v7 WATCH tier, gradual harmful shifts, 6 unseen datasets")
    st.metric("Median warning lead", f"{c1['median_lead']:.0f} batches", border=True,
              help="Batches between the first warning and the accuracy drop")
    st.metric("False alerts", f"{c1['false_alerts']:.1f} per 100 batches", border=True,
              help="On clean batches; budget is 5 per 100")
    st.metric("Harmless-drift alarms with the harm tier", f"{d3['harm_harmless']:.0%}", border=True,
              help=f"Versus {d3['watch_harmless']:.0%} for the drift watch tier alone (v8)")
    st.metric("Harm separation (AUROC)", f"{d4['auroc_harm']:.2f}", border=True,
              help=f"Harmful vs harmless shifts; drift signals alone reach {d4['auroc_watch']:.2f}")

timeline = pd.DataFrame([
    {"Version": "v1", "What was tested": "Risk calibrator on synthetic episodes", "Outcome": "Failure",
     "Lesson": "Scenarios had no lead time and the risk target was inverted"},
    {"Version": "v2–v4", "What was tested": "Redesigned scenarios, real streams, delay-anchored estimator",
     "Outcome": "Mixed", "Lesson": "Works in controlled data; drift scores do not transfer across datasets"},
    {"Version": "v6", "What was tested": "Pre-registered benchmark on real data with controlled shifts",
     "Outcome": r["v6"]["overall"].title(), "Lesson": "Thresholds in raw units broke on unseen datasets"},
    {"Version": "v7", "What was tested": "Stream-normalized drift watch tier",
     "Outcome": v7["overall"].title(), "Lesson": "Early drift warning works and transfers; it cannot tell harmful from harmless"},
    {"Version": "v8", "What was tested": "Harm tier from the labeled reference set",
     "Outcome": v8["overall"].title(), "Lesson": "Harmless alarms cut from 97% to 34%; AUROC 0.78, just below the 0.80 target"},
])

left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.subheader("How the results were reached", anchor=False)
        st.dataframe(timeline, hide_index=True, alt="Experiment versions and their outcomes")
with right:
    with st.container(border=True):
        st.subheader("What the system can claim", anchor=False)
        st.markdown(
            ":material/check_circle: Warns about drift **before** accuracy drops on unseen datasets, "
            "within the false-alert budget.\n\n"
            ":material/check_circle: Ranks harmful drift clearly better than drift statistics "
            f"(AUROC {d4['auroc_harm']:.2f} vs {d4['auroc_watch']:.2f}).\n\n"
            ":material/cancel: Does not reliably separate harmful from harmless drift "
            f"({d3['harm_harmless']:.0%} of harmless drifts still alert).\n\n"
            ":material/cancel: Cannot see label changes (concept drift) before labels arrive.\n\n"
            ":material/cancel: Fails on natural streams that are already drifting when monitoring starts."
        )
