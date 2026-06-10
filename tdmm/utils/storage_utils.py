from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def save_df(
    df: pd.DataFrame,
    path_stem: str | Path,
    *,
    export_csv: bool = False,
    csv_max_rows: int = 200_000,
    csv_encoding: str = "utf-8-sig",
) -> Path:
    """Save dataframe as Parquet (preferred) and optionally as CSV.

    `path_stem` may be either a stem without extension or a concrete path.

    Notes for Windows users:
    - Parquet is compact and fast, but not always convenient to open manually.
    - CSV is convenient for Word/Excel imports, but can be huge.
      Therefore CSV export is size-limited via `csv_max_rows`.

    Returns the *primary* path written (Parquet if available, else CSV).
    """

    p = Path(path_stem)
    stem = p.with_suffix("") if p.suffix in [".parquet", ".csv"] else p
    stem.parent.mkdir(parents=True, exist_ok=True)

    wrote: Optional[Path] = None

    # 1) parquet if possible
    if _parquet_available():
        out_parquet = stem.with_suffix(".parquet")
        df.to_parquet(out_parquet, index=False)
        wrote = out_parquet

    # 2) csv (always if parquet is unavailable; optionally otherwise)
    need_csv = (wrote is None) or export_csv
    if need_csv:
        if len(df) <= int(csv_max_rows):
            out_csv = stem.with_suffix(".csv")
            df.to_csv(out_csv, index=False, encoding=csv_encoding)
            if wrote is None:
                wrote = out_csv
        else:
            # For huge frames write a sample, so the user still has something readable.
            out_sample = stem.with_suffix(".sample.csv")
            df.head(50_000).to_csv(out_sample, index=False, encoding=csv_encoding)
            if wrote is None:
                wrote = out_sample

    assert wrote is not None
    return wrote


def load_df(path_stem: str | Path) -> pd.DataFrame:
    """Load dataframe from Parquet if present, otherwise CSV."""
    p = Path(path_stem)
    stem = p.with_suffix("") if p.suffix in [".parquet", ".csv"] else p
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, encoding="utf-8-sig")
    # fallback: sample
    sample = stem.with_suffix(".sample.csv")
    if sample.exists():
        return pd.read_csv(sample, encoding="utf-8-sig")
    raise FileNotFoundError(f"Neither {parquet} nor {csv} exists.")
