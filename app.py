import streamlit as st

from components.theme import load_theme
from components.layout.header import render_header
from components.layout.navigation import render_navigation

from components.chat.interface import render_chat
from components.pages.architecture import render_architecture
from components.pages.about import render_about


st.set_page_config(
    page_title="AI Knowledge Orchestrator",
    page_icon="🤖",
    layout="wide",
)

load_theme()

# ---------------------------
# Navigation State
# ---------------------------

if "page" not in st.session_state:
    st.session_state.page = "Chat"

render_header()

left, right = st.columns([1, 3], gap="large")

with left:
    render_navigation()

with right:

    if st.session_state.page == "Chat":
        render_chat()

    elif st.session_state.page == "Architecture":
        render_architecture()

    elif st.session_state.page == "About":
        render_about()