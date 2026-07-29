import streamlit as st


def render_hero():

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:

        st.markdown(
            "<div style='text-align:center;color:#64748B;font-size:64px;'>🤖</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h1 style='text-align:center;color:#1E4174;'>AI Knowledge Assistant</h1>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div style="text-align:center;
            font-size:20px;
            color:#64748B;
            max-width:700px;
            margin:auto;">
Ask questions, summarize documents, and discover insights from your PDFs.
</div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")