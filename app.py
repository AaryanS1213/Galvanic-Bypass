"""Streamlit app for Galvanic Bypass. Run: streamlit run app.py"""

import streamlit as st

from gb.config import assumptions_table, load_config

st.set_page_config(page_title="Galvanic Bypass", layout="wide")
st.title("Galvanic Bypass")

cfg = load_config()
st.subheader("Assumptions")
table = assumptions_table(cfg)
table["value"] = table["value"].astype(str)
st.dataframe(table, use_container_width=True, hide_index=True)
