"""ML Model Drift Early-Warning System: results and live-monitor dashboard.

    streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Drift early-warning system", page_icon=":material/monitoring:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:"),
        st.Page("app_pages/live_monitor.py", title="Live monitor", icon=":material/monitor_heart:"),
        st.Page("app_pages/verdicts.py", title="Verdicts", icon=":material/fact_check:"),
        st.Page("app_pages/ablations.py", title="Why it works", icon=":material/science:"),
    ],
    position="top",
)
st.title(page.title, anchor=False)
page.run()
