import streamlit as st


def render_header():

    left, right = st.columns([7, 3], vertical_alignment="center")

    with left:
        st.markdown(
            """
            <div class="app-title">
                🤖 AI Knowledge Orchestrator
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(
                "Chat",
                key="nav_chat",
                use_container_width=True,
                type="primary" if st.session_state.page == "Chat" else "secondary",
            ):
                st.session_state.page = "Chat"
                st.rerun()

        with c2:
            if st.button(
                "Architecture",
                key="nav_arch",
                use_container_width=True,
                type="primary" if st.session_state.page == "Architecture" else "secondary",
            ):
                st.session_state.page = "Architecture"
                st.rerun()

        with c3:
            if st.button(
                "About",
                key="nav_about",
                use_container_width=True,
                type="primary" if st.session_state.page == "About" else "secondary",
            ):
                st.session_state.page = "About"
                st.rerun()

    st.markdown("<div class='header-divider'></div>", unsafe_allow_html=True)