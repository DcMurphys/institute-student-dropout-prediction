"""Project paths and app-wide constants shared by every layer.

Paths are resolved from the repository root, so the app works no matter
which folder ``streamlit run`` is started from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

# ===== Model artifacts (exported by the notebook, Section 8.1) =====
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "trained_models"
PIPELINE_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
TARGET_ENCODER_PATH = MODEL_DIR / "encoder_target.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"

# ===== Data =====
DATA_DIR = PROJECT_ROOT / "data"
STUDENT_DATA_PATH = DATA_DIR / "processed" / "student.csv"  # notebook Section 5.2
REFERENCE_DIR = DATA_DIR / "reference"
LABEL_MAPPINGS_PATH = REFERENCE_DIR / "label_mappings.json"
MODEL_PERFORMANCE_PATH = REFERENCE_DIR / "model_performance.json"

# ===== Brand =====
BRAND_DIR = APP_DIR / "assets" / "brand"
LOGO_PATH = BRAND_DIR / "murphys_polytechnic_logo.svg"
SYMBOL_PATH = BRAND_DIR / "murphys_polytechnic_symbol.svg"
INSTITUTION_NAME = "Murphys Polytechnic Institute"
APP_NAME = "Student Success Early Warning"
FOOTER_TEXT = (
    f"Portfolio prototype · {INSTITUTION_NAME} is a fictional institution · "
    "Data: Predict Students' Dropout and Academic Success (Realinho et al., 2021), "
    "used as a proxy for institutional records."
)

# ===== Domain constants =====
TARGET = "Status"
POSITIVE_CLASS = "Dropout"
STATUS_ORDER = ("Dropout", "Enrolled", "Graduate")
MIN_GROUP_SIZE = 50  # smaller groups give unstable rates (notebook Section 4.4.10)
MAX_UPLOAD_ROWS = 5_000

# Palette tokens (Murphys design system, shared with the other portfolio apps)
PALETTE = {
    "brand": "#FF6A13",
    "brand-dark": "#C2410C",
    "ink": "#1B1B1F",
    "slate": "#475467",
    "surface": "#FFFFFF",
    "surface-tint": "#FFF4EC",
    "border": "#E7E7EA",
    "context-grey": "#98A2B3",
    "light-grey": "#D0D5DD",
    "status-dropout": "#B42318",
    "status-enrolled": "#F79009",
    "status-enrolled-text": "#B54708",
    "status-graduate": "#067647",
    "info": "#2E5AAC",
}
