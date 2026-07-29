from pathlib import Path
import streamlit as st


def load_theme():

    css = Path("components/styles.css").read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )