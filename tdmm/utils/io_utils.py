from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

def read_text_with_fallback(path: str | Path, encodings: Iterable[str]) -> Tuple[str, str]:
    """
    Read a text file trying several encodings.

    Returns: (text, encoding_used)
    """
    p = Path(path)
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return p.read_text(encoding=enc), enc
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read {p} with encodings={list(encodings)}; last_error={last_err}") from last_err

def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding=encoding)
