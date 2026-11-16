from datetime import datetime

import streamlit as st

from app.llm.citations import cited_sources


def build_transcript(messages, exported_at=None):
    """
    The conversation as a Markdown document, with the sources
    each answer cited.
    """

    exported_at = exported_at or datetime.now()

    lines = [
        "# Chat export",
        "",
        f"Exported {exported_at:%Y-%m-%d %H:%M}",
    ]

    for message in messages:

        if message["role"] == "user":

            lines += ["", "## You", "", message["content"]]

            continue

        lines += ["", "## Assistant", "", message["content"]]

        cited = cited_sources(
            message["content"],
            message.get("sources", []),
        )

        if cited:

            lines += ["", "Sources:", ""]

            lines += [
                f"- [{number}] {source.location}"
                for number, source in cited
            ]

    return "\n".join(lines) + "\n"


def render_export():

    messages = st.session_state.get("messages", [])

    if not messages:
        return

    st.divider()

    st.subheader("💬 Conversation")

    now = datetime.now()

    st.download_button(
        "Download chat",
        data=build_transcript(messages, now),
        file_name=f"chat-{now:%Y%m%d-%H%M}.md",
        mime="text/markdown",
        icon=":material/download:",
        use_container_width=True,
    )
