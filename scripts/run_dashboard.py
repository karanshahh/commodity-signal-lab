#!/usr/bin/env python3
"""Launch Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and src on path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    app_path = root / "src" / "signal_lab" / "dashboard" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.port=8501"]
    stcli.main()
