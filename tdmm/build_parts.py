from __future__ import annotations

"""Build meaningful corpus segmentation (books + parts) from explicit headings.

The user's corpus for «Тихий Дон» contains headings:

  • «Книга первая», «Книга вторая», «Книга третья», «Книга четвертая/четвёртая»
  • «Часть первая» … «Часть восьмая»

These boundaries are required for philologically valid dynamics (not quartiles).
The implementation is line-based (to reliably catch headings), then mapped to
`sent_id` via razdel sentence offsets.

Defaults:
  • heading sentences are excluded from segments
  • book segments start from the first part of the book (epigraph/prologue is
    excluded) when parts are detected

CLI:
  python -m tdmm.build_parts --corpus data\\corpus\\quiet_don.txt \
      --out-books data\\segmentation\\books.csv \
      --out-parts data\\segmentation\\parts.csv
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from tdmm.utils.io_utils import read_text_with_fallback
from tdmm.utils.text_utils import sentenize_text


_BOOK_WORD2ID: Dict[str, int] = {
    "первая": 1,
    "вторая": 2,
    "третья": 3,
    "четвертая": 4,
    "четвёртая": 4,
}

_PART_WORD2ID: Dict[str, int] = {
    "первая": 1,
    "вторая": 2,
    "третья": 3,
    "четвертая": 4,
    "четвёртая": 4,
    "пятая": 5,
    "шестая": 6,
    "седьмая": 7,
    "восьмая": 8,
}


_RE_BOOK_LINE = re.compile(
    r"^\s*Книга\s+(первая|вторая|третья|четвертая|четвёртая)\s*[.]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

_RE_PART_LINE = re.compile(
    r"^\s*Часть\s+(первая|вторая|третья|четвертая|четвёртая|пятая|шестая|седьмая|восьмая)\s*[.]*\s*$",
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class _Heading:
    kind: str  # BOOK|PART
    idx: int   # book_id or part_id
    name: str  # output name
    char_pos: int


def _scan_line_headings(text: str) -> List[_Heading]:
    """Scan raw text by lines and collect heading positions (character offsets)."""
    headings: List[_Heading] = []
    pos = 0
    for line in text.splitlines(True):
        raw = line.strip()
        if not raw:
            pos += len(line)
            continue

        m = _RE_BOOK_LINE.match(raw)
        if m:
            w = m.group(1).lower()
            bid = _BOOK_WORD2ID.get(w, -1)
            if bid != -1:
                headings.append(_Heading("BOOK", bid, f"Книга {w}", pos))
            pos += len(line)
            continue

        m = _RE_PART_LINE.match(raw)
        if m:
            w = m.group(1).lower()
            pid = _PART_WORD2ID.get(w, -1)
            if pid != -1:
                headings.append(_Heading("PART", pid, f"Часть {w}", pos))
            pos += len(line)
            continue

        pos += len(line)

    headings.sort(key=lambda h: h.char_pos)
    return headings


def _sent_offsets(text: str) -> List[Tuple[int, int]]:
    return [(s.start, s.stop) for s in sentenize_text(text)]


def _charpos_to_sent_id(offsets: List[Tuple[int, int]], pos: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s0, s1 = offsets[mid]
        if s0 <= pos < s1:
            return mid
        if pos < s0:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(offsets) - 1, lo - 1))


def _fallback_equal(n_sents: int, n_segments: int, prefix: str) -> pd.DataFrame:
    if n_sents <= 0:
        return pd.DataFrame(columns=[f"{prefix}_id", f"{prefix}_name", "start_sent", "end_sent"])
    n_segments = max(1, min(int(n_segments), int(n_sents)))
    q, rem = divmod(int(n_sents), n_segments)
    rows = []
    start = 0
    for i in range(1, n_segments + 1):
        size = q + (1 if i <= rem else 0)
        end = min(n_sents - 1, start + size - 1)
        rows.append({
            f"{prefix}_id": i,
            f"{prefix}_name": f"{prefix.capitalize()} {i}",
            "start_sent": int(start),
            "end_sent": int(end),
        })
        start = end + 1
    return pd.DataFrame(rows)


def build_books_and_parts(
    corpus_path: str,
    *,
    out_books_csv: Optional[str] = None,
    out_parts_csv: Optional[str] = None,
    encoding_fallbacks: Optional[List[str]] = None,
    exclude_headers: bool = True,
    exclude_epigraph: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (books_df, parts_df) and optionally write CSVs."""
    text, _ = read_text_with_fallback(corpus_path, encodings=encoding_fallbacks)
    offsets = _sent_offsets(text)
    n_sents = len(offsets)

    headings = _scan_line_headings(text)
    book_heads: List[Tuple[int, str, int]] = []
    part_heads: List[Tuple[int, str, int]] = []
    for h in headings:
        sid = _charpos_to_sent_id(offsets, h.char_pos)
        if h.kind == "BOOK":
            book_heads.append((h.idx, h.name, sid))
        else:
            part_heads.append((h.idx, h.name, sid))

    # Deduplicate (keep first occurrence per id).
    def _dedup(items: List[Tuple[int, str, int]]) -> List[Tuple[int, str, int]]:
        seen = set()
        out = []
        for idx, name, sid in sorted(items, key=lambda x: x[2]):
            if idx in seen:
                continue
            seen.add(idx)
            out.append((idx, name, sid))
        return out

    book_heads = _dedup(book_heads)
    part_heads = _dedup(part_heads)

    if len(book_heads) < 2:
        books_df = _fallback_equal(n_sents, 4, "book")
        parts_df = _fallback_equal(n_sents, 8, "part")
        if out_books_csv:
            Path(out_books_csv).parent.mkdir(parents=True, exist_ok=True)
            books_df.to_csv(out_books_csv, index=False)
        if out_parts_csv:
            Path(out_parts_csv).parent.mkdir(parents=True, exist_ok=True)
            parts_df.to_csv(out_parts_csv, index=False)
        return books_df, parts_df

    book_heads.sort(key=lambda x: x[2])
    part_heads.sort(key=lambda x: x[2])

    # Find first part after a book header, before next book.
    def _first_part_after(book_sid: int, next_book_sid: int) -> Optional[int]:
        for _, _, psid in part_heads:
            if book_sid < psid < next_book_sid:
                return psid
        return None

    # Books
    book_rows = []
    for i, (bid, bname, b_sid) in enumerate(book_heads):
        next_b_sid = book_heads[i + 1][2] if i + 1 < len(book_heads) else n_sents

        start_sid = b_sid
        if exclude_epigraph:
            fp = _first_part_after(b_sid, next_b_sid)
            if fp is not None:
                start_sid = fp
        if exclude_headers:
            start_sid = min(n_sents - 1, start_sid + 1)
        end_sid = max(0, next_b_sid - 1)

        book_rows.append(
            {
                "book_id": int(bid),
                "book_name": str(bname),
                "start_sent": int(start_sid),
                "end_sent": int(end_sid),
            }
        )

    books_df = pd.DataFrame(book_rows).sort_values("book_id")

    # Parts + mapping to current book
    def _book_for_sid(sid: int) -> Tuple[int, str]:
        current = book_heads[0]
        for b in book_heads:
            if b[2] <= sid:
                current = b
            else:
                break
        return int(current[0]), str(current[1])

    part_rows = []
    for i, (pid, pname, p_sid) in enumerate(part_heads):
        next_p_sid = part_heads[i + 1][2] if i + 1 < len(part_heads) else n_sents
        # Cap by next book header
        for _, _, b_sid in book_heads:
            if p_sid < b_sid < next_p_sid:
                next_p_sid = b_sid
                break
        start_sid = p_sid + 1 if exclude_headers else p_sid
        start_sid = min(n_sents - 1, start_sid)
        end_sid = max(0, next_p_sid - 1)
        bid, bname = _book_for_sid(p_sid)
        part_rows.append(
            {
                "part_id": int(pid),
                "part_name": str(pname),
                "book_id": bid,
                "book_name": bname,
                "start_sent": int(start_sid),
                "end_sent": int(end_sid),
            }
        )

    parts_df = pd.DataFrame(part_rows).sort_values("part_id") if part_rows else _fallback_equal(n_sents, 8, "part")

    if out_books_csv:
        Path(out_books_csv).parent.mkdir(parents=True, exist_ok=True)
        books_df.to_csv(out_books_csv, index=False)
    if out_parts_csv:
        Path(out_parts_csv).parent.mkdir(parents=True, exist_ok=True)
        parts_df.to_csv(out_parts_csv, index=False)

    return books_df, parts_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    # Backward-compatible alias: older versions used only --out (a single CSV).
    # We treat it as the *books* segmentation output and also write parts.csv
    # рядом, если пользователь не указал --out-parts.
    ap.add_argument("--out", default=None, help="Legacy: output books segmentation CSV")
    ap.add_argument("--out-books", default="data/segmentation/books.csv")
    ap.add_argument("--out-parts", default="data/segmentation/parts.csv")
    ap.add_argument("--include-headers", action="store_true")
    ap.add_argument("--include-epigraph", action="store_true")
    args = ap.parse_args()

    out_books = args.out_books
    out_parts = args.out_parts
    if args.out:
        out_books = args.out
        # if user didn't override out_parts explicitly, place it рядом
        if "--out-parts" not in " ".join(__import__("sys").argv):
            out_parts = str(Path(out_books).with_name("parts.csv"))

    build_books_and_parts(
        args.corpus,
        out_books_csv=out_books,
        out_parts_csv=out_parts,
        encoding_fallbacks=["utf-8", "utf-8-sig", "cp1251"],
        exclude_headers=not args.include_headers,
        exclude_epigraph=not args.include_epigraph,
    )
    print(f"OK: wrote {out_books} and {out_parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
