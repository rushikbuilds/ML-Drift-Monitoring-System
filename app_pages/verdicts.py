import streamlit as st

from dashboard.app_data import doc_text, load_results


@st.cache_data(max_entries=1)
def results():
    return load_results()


r = results()
docs = {
    f"v8 harm tier ({r['v8']['overall'].lower()})": "v8_verdict.md",
    f"v7 drift watch ({r['v7']['overall'].lower()})": "v7_verdict.md",
    f"v6 benchmark ({r['v6']['overall'].lower()})": "verdict.md",
}
st.caption("Every verdict was computed by a script from a one-time run, with hypotheses and thresholds "
           "frozen beforehand. Earlier verdicts are never edited.")
choice = st.segmented_control("Experiment", list(docs), default=list(docs)[0], key="verdict_choice") or list(docs)[0]
overall = {"v8_verdict.md": r["v8"]["overall"], "v7_verdict.md": r["v7"]["overall"], "verdict.md": r["v6"]["overall"]}[docs[choice]]
badge_color = {"SUCCESS": "green", "PARTIAL SUCCESS": "orange", "FAILURE": "red"}[overall]
st.badge(overall.title(), color=badge_color, icon=":material/gavel:")
with st.container(border=True):
    st.markdown(doc_text(docs[choice]))
