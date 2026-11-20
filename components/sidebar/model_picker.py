import streamlit as st

from app.core.config import CLAUDE_MODELS, DEFAULT_CLAUDE_MODEL
from app.llm.llm import available_models, create_llm, has_api_key


@st.cache_resource(show_spinner=False)
def get_llm(model):
    """
    One client per model, shared between sessions.
    """
    return create_llm(model)


def selected_llm():
    """
    The Claude model this browser session has chosen.
    """

    return get_llm(
        st.session_state.get("model_choice") or DEFAULT_CLAUDE_MODEL
    )


def render_model_picker():

    st.subheader("🤖 Model")

    models = available_models()

    current = st.session_state.get("model_choice")

    if current not in models:
        current = models[0]

    st.session_state.model_choice = st.selectbox(
        "Model",
        models,
        index=models.index(current),
        format_func=CLAUDE_MODELS.get,
        label_visibility="collapsed",
        disabled=not has_api_key(),
    )
