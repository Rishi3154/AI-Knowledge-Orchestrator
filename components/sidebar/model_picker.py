import streamlit as st

from app.core.config import (
    CLAUDE_MODELS,
    DEFAULT_CLAUDE_MODEL,
    anthropic_api_key,
)
from app.core.code_version import code_version
from app.llm.llm import available_models, create_llm, has_api_key


@st.cache_resource(max_entries=6, show_spinner=False)
def get_llm(model, api_key, version):
    """
    One client per model, shared between sessions. The key and the
    code version are part of the cache key, so a replaced key or a
    code update gets a new client.
    """
    return create_llm(model)


def selected_llm():
    """
    The Claude model this browser session has chosen.
    """

    return get_llm(
        st.session_state.get("model_choice") or DEFAULT_CLAUDE_MODEL,
        anthropic_api_key(),
        code_version(),
    )


def render_model_picker():

    st.subheader("🤖 Model")

    models = available_models()

    current = st.session_state.get("model_choice")

    if current not in models:
        current = models[0]

    # Nothing to choose between: just name the model.
    if len(models) == 1:

        st.session_state.model_choice = current

        st.success(CLAUDE_MODELS[current])

        return

    st.session_state.model_choice = st.selectbox(
        "Model",
        models,
        index=models.index(current),
        format_func=CLAUDE_MODELS.get,
        label_visibility="collapsed",
        disabled=not has_api_key(),
    )
