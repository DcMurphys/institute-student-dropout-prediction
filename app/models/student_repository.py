"""Load student records and reference tables; translate codes into labels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import TARGET


def load_students(path: Path) -> pd.DataFrame:
    """Read the dashboard dataset exported by the notebook (original units, codes).

    Raises:
        FileNotFoundError: If the CSV has not been exported yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebook Section 5.2 to export it."
        )
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"{path.name} has no '{TARGET}' column.")
    return data


def load_json(path: Path) -> dict:
    """Read a reference JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_label_mappings(path: Path) -> dict[str, dict[int, str]]:
    """Read code-to-label tables; keys starting with ``_`` are documentation."""
    raw = load_json(path)
    return {
        column: {int(code): label for code, label in table.items()}
        for column, table in raw.items()
        if not column.startswith("_")
    }


def code_label(code: int, column: str, mappings: dict[str, dict[int, str]]) -> str:
    """Readable label for one code (falls back to ``Code <n>``)."""
    return mappings.get(column, {}).get(int(code), f"Code {code}")


def decode_labels(
    data: pd.DataFrame, mappings: dict[str, dict[int, str]]
) -> pd.DataFrame:
    """Return a copy where every mapped code column shows readable labels."""
    decoded = data.copy()
    for column, table in mappings.items():
        if column in decoded.columns:
            decoded[column] = decoded[column].map(
                lambda code, t=table: t.get(code, f"Code {code}")
            )
    return decoded
