import altair as alt
import pandas as pd
import streamlit as st

from dashboard.app_data import load_results


@st.cache_data(max_entries=1)
def results():
    return load_results()


ab = results()["ablations"]
READABLE = {"primary_system": "v6 risk calibrator", "fixed_dews_score": "DEWS drift score (raw)",
            "statistical_only": "Statistical drift (raw)", "confidence_only": "Confidence only", "mean_ks": "Mean KS (raw)",
            "label_trend": "Waiting for labels", "delay_aware_cbpe": "Delay-aware CBPE",
            "v7 WATCH (normalized)": "v7 drift watch", "v8 HARM (accuracy units)": "v8 harm tier"}
st.caption("Post-hoc analyses that explain the verdicts. They do not change them.")

left, right = st.columns(2)
with left:
    with st.container(border=True, height="stretch"):
        st.subheader("Thresholds only transfer in comparable units", anchor=False)
        st.caption("False alerts per 100 clean batches with a threshold chosen on development data. Budget: 5.")
        rows = [{"method": READABLE.get(m, m), "data": d, "false alerts": v[k]}
                for m, v in ab["A_threshold_transfer"].items()
                for k, d in (("development", "Development"), ("held_out", "Unseen datasets"))]
        frame = pd.DataFrame(rows)
        bars = alt.Chart(frame).mark_bar().encode(
            y=alt.Y("method:N", title=None, axis=alt.Axis(labelLimit=220), sort=[READABLE.get(m, m) for m in ab["A_threshold_transfer"]]),
            x=alt.X("false alerts:Q", title="False alerts per 100 clean batches"),
            yOffset="data:N", color=alt.Color("data:N", title=None),
            tooltip=["method", "data", alt.Tooltip("false alerts:Q", format=".1f")])
        budget = alt.Chart(pd.DataFrame({"x": [5]})).mark_rule(color="firebrick", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(bars + budget, alt="False alerts on development versus unseen datasets for each method")
        st.markdown("Scores in raw units (drift statistics) break the budget 4–10× on new datasets. "
                    "Scores relative to the stream's own normal, or in accuracy points, hold it.")
with right:
    with st.container(border=True, height="stretch"):
        st.subheader("Drift signals and harm signals behave differently", anchor=False)
        st.caption("Each signal used alone as the drift watch tier, on unseen data.")
        names = {"drift_score": "DEWS drift score", "mean_ks": "Mean KS", "rws": "Relevance-weighted shift",
                 "disagreement": "Model disagreement", "confidence_drop": "Confidence drop", "entropy": "Entropy",
                 "positive_rate_shift": "Positive-rate shift"}
        rows = [{"signal": names[k.replace("only ", "")], "warned before drop": v["t1_gradual"],
                 "harmless drifts alerting": v["harmless_alert_share"], "harm AUROC": v["harm_auroc"]}
                for k, v in ab["B_watch_signals"].items() if k.startswith("only ")]
        frame = pd.DataFrame(rows)
        frame["family"] = frame["signal"].map(lambda s: "Model behaviour" if s in ("Model disagreement", "Confidence drop", "Entropy") else "Input shift")
        st.altair_chart(alt.Chart(frame).mark_circle(size=160).encode(
            x=alt.X("harmless drifts alerting:Q", title="Harmless drifts that alert", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("warned before drop:Q", title="Harmful drifts warned early", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("family:N", title=None, scale=alt.Scale(domain=["Input shift", "Model behaviour"], range=["#4c9be8", "#f28e2b"])),
            tooltip=["signal", alt.Tooltip("warned before drop:Q", format=".2f"),
                     alt.Tooltip("harmless drifts alerting:Q", format=".2f"), alt.Tooltip("harm AUROC:Q", format=".2f")]),
            alt="Early warning versus harmless alarms for each drift signal")
        st.markdown("Input-shift signals warn early but alert on almost every harmless drift. "
                    "Model-behaviour signals are quieter and more harm-aware, but warn later.")

left, right = st.columns(2)
with left:
    with st.container(border=True, height="stretch"):
        st.subheader("Harm comes from where the model is wrong", anchor=False)
        rows = [{"Data": k, "Shifts": v["shift_episodes"], "Harmful": f"{v['harmful_share']:.0%}",
                 "Important feature shifted": f"{v['harmful_share_important_feature']:.0%}",
                 "Unimportant feature shifted": f"{v['harmful_share_unimportant_feature']:.0%}"}
                for k, v in ab["E_harm_rates"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Share of shifts that were harmful")
        st.markdown("Only about one shift in five hurts the model, and shifting an important feature is "
                    "barely more harmful. That is why the harm tier looks at where the model makes mistakes "
                    "on the labeled reference data instead.")
with right:
    with st.container(border=True, height="stretch"):
        st.subheader("Why natural streams fail", anchor=False)
        f = ab["F_natural_streams"]
        with st.container(horizontal=True):
            st.metric("Median accuracy event", f"batch {f['event_onset_quartiles'][1]:.0f}", border=True)
            st.metric("Events right after burn-in", f"{f['events_within_5_batches_after_burn_in']} of {f['windows_with_event']}", border=True)
        st.markdown("On natural streams the drift is already under way when monitoring starts, so the burn-in "
                    "period that defines \"normal\" is not normal. Stream normalization needs a stable start.")
