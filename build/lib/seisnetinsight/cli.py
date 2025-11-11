"""Command line entry points."""

from __future__ import annotations

from pathlib import Path


def run_app() -> None:
    """Launch the Streamlit application."""
    try:
        from streamlit.web import bootstrap
    except ImportError:  # pragma: no cover
        raise SystemExit("Streamlit is required to run the SeisNetInsight app.")

    script_path = Path(__file__).with_name("streamlit_app.py")
    # Pass False for is_hello to launch a regular Streamlit app session.
    bootstrap.run(str(script_path), False, [], {})
