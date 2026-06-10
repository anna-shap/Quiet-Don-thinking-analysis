from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

def read_csv_flexible(path: str | Path, encodings: Iterable[str], seps: Iterable[str]) -> pd.DataFrame:
    """
    Try reading a CSV with multiple encodings and separators.
    """
    p = Path(path)
    last_err: Optional[Exception] = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(p, encoding=enc, sep=sep)
            except Exception as e:  # noqa: BLE001
                last_err = e
    # fallback: pandas auto-sep with python engine (slower)
    for enc in encodings:
        try:
            return pd.read_csv(p, encoding=enc, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read CSV {p}; last_error={last_err}") from last_err

def safe_to_csv(df: pd.DataFrame, path: str | Path, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding=encoding)
