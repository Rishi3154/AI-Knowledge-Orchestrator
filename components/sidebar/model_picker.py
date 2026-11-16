import streamlit as st

from app.llm.llm import (
    CLAUDE,
    OLLAMA,
    OPENAI,
    available_models,
    create_llm,
    default_model,
)


@st.cache_data(ttl=60, show_spinner=False)
def list_models():
    """
    Models that can be chosen. Cached briefly so the list isn't
    fetched from Ollama on every click.
    """
    return available_models()


@st.cache_resource(show_spinner=False)
def get_llm(provider, model):
    """
    One client per model, shared between sessions.
    """
    return create_llm(provider, model)


def model_label(choice):

    provider, model = choice

    if provider == OPENAI:
        return f"{model} (OpenAI, cloud)"

    if provider == CLAUDE:
        return f"{model} (Anthropic, cloud)"

    return f"{model} (local)"


def selected_llm():
    """
    The LLM this browser session has chosen.
    """

    choice = st.session_state.get("model_choice") or default_model()

    return get_llm(*choice)


def render_model_picker():

    st.subheader("🤖 Model")

    models = list_models()

    current = st.session_state.get("model_choice")

    # The chosen model may have been removed since it was picked.
    if current not in models:
        current = models[0]

    if len(models) == 1:

        st.session_state.model_choice = current

        st.success(model_label(current))

        return

    choice = st.selectbox(
        "Model",
        models,
        index=models.index(current),
        format_func=model_label,
        label_visibility="collapsed",
    )

    st.session_state.model_choice = choice

    if choice[0] != OLLAMA and default_model()[0] == OLLAMA:

        company = "Anthropic" if choice[0] == CLAUDE else "OpenAI"

        st.caption(
            f"Questions and the retrieved passages are sent to {company}."
        )
