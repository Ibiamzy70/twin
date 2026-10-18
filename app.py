"""
Digital Twin - application entry point.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from components.sidebar import render_sidebar
from state import initialize_state
from views import assistant, calculator, dashboard, settings, tasks

# ---------------------------------------------------------------------------
# 1. Page configuration
#    Must be the first Streamlit command executed. The imports above are fine
#    as long as none of those modules call st.* at import time.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 2. Theme loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
THEME_PATH = BASE_DIR / "styles" / "theme.css"


def load_theme() -> None:
    """Load the local stylesheet. Only use with trusted CSS files you control."""
    try:
        css = THEME_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(f"Theme file not found: {THEME_PATH}")
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_theme()


# ---------------------------------------------------------------------------
# 3. Session state (created once, never reset by views)
# ---------------------------------------------------------------------------
initialize_state()


# ---------------------------------------------------------------------------
# 4. Display preferences (Settings -> Preferences)
#    Injected on every run, on every page, after the theme so they can override it.
#    Toggling a preference reruns the app, so the change shows up immediately.
# ---------------------------------------------------------------------------
settings.apply_display_preferences()


# ---------------------------------------------------------------------------
# 5. Route-to-view mapping
# ---------------------------------------------------------------------------
VIEWS = {
    "overview": dashboard.render,
    "assistant": assistant.render,
    "calculator": calculator.render,
    "tasks": tasks.render,
    "settings": settings.render,
}


# ---------------------------------------------------------------------------
# 6. Application entry point
# ---------------------------------------------------------------------------
def main() -> None:
    route_id = render_sidebar()

    view = VIEWS.get(route_id)
    if view is None:
        st.error(f"Unknown page route: {route_id}")
        return

    view()


if __name__ == "__main__":
    main()
