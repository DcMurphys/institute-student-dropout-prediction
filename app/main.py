"""Streamlit entry point: student dropout early-warning app.

Run from the repository root:
    streamlit run app/main.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # makes the ``app`` package importable
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

from app.controllers.router import run  # noqa: E402

run()
