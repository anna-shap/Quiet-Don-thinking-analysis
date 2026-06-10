from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

QUOTE_OPEN = "«"
QUOTE_CLOSE = "»"


@dataclass
class Window:
    start: int
    end: int


@dataclass
class TextSpan:
    text: str
    start: int
    stop: int


HEADING_RE = re.compile(
    r"^\s*(?:"
    r"книга\s+(?:первая|вторая|третья|четв[её]ртая|[ivxlcdm]+|\d+)"
    r"|часть\s+(?:первая|вторая|третья|четв[её]ртая|пятая|шестая|седьмая|восьмая|[ivxlcdm]+|\d+)"
    r"|глава\s+(?:[ivxlcdm]+|\d+|[а-яё-]+)"
    r"|том\s+(?:[ivxlcdm]+|\d+|[а-яё-]+)"
    r")\s*[.!?…—-]*\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)


def split_headings_from_text(text: str, *, remove_headings: bool = True) -> Tuple[str, List[Dict[str, object]]]:
    """Return analytical text and heading rows detected line-by-line.

    The corpus segmentation files remain authoritative; this function only prevents
    structural headings like ``Книга первая`` from being glued to the first
    analytical sentence and contaminating NLP output.
    """
    headings: List[Dict[str, object]] = []
    out_lines: List[str] = []
    char_pos = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped and HEADING_RE.match(stripped):
            htype = "book" if stripped.lower().startswith("книга") else ("part" if stripped.lower().startswith("часть") else "heading")
            headings.append({
                "heading_id": len(headings) + 1,
                "line_no": line_no,
                "char_pos_approx": char_pos,
                "heading_type": htype,
                "heading_text": stripped,
                "removed_from_analysis": bool(remove_headings),
            })
            if remove_headings:
                out_lines.append("")
            else:
                out_lines.append(raw)
        else:
            out_lines.append(raw)
        char_pos += len(raw) + 1
    return "\n".join(out_lines), headings


def looks_like_direct_speech(sentence: str) -> bool:
    if QUOTE_OPEN in sentence and QUOTE_CLOSE in sentence:
        return True
    if re.match(r"^\s*[—-]\s+\S", sentence):
        return True
    return False


def contains_any(text: str, variants: List[str]) -> bool:
    return any(v in text for v in variants)


def extract_proper_names(sentence: str) -> List[str]:
    tokens = re.findall(r"[А-ЯЁ][а-яё]+", sentence)
    if not tokens:
        return []
    out: List[str] = []
    for i, t in enumerate(tokens):
        if i == 0 and len(sentence.lstrip()) > 0:
            if re.search(r"(ов|ев|ин|ына|ова|ева|ина)$", t) or "Ё" in t:
                out.append(t)
        else:
            out.append(t)
    return out


def sentence_window(idx: int, total: int, radius: int) -> Window:
    return Window(start=max(0, idx - radius), end=min(total - 1, idx + radius))


def sentenize_text(text: str) -> List[TextSpan]:
    """Use razdel when available; otherwise a conservative regex fallback."""
    try:
        from razdel import sentenize  # type: ignore
        return [TextSpan(s.text, s.start, s.stop) for s in sentenize(text)]
    except Exception:
        spans: List[TextSpan] = []
        pattern = re.compile(r"[^.!?…]+(?:[.!?…]+|$)", re.UNICODE)
        for m in pattern.finditer(text):
            s = m.group(0).strip()
            if not s:
                continue
            # reconstruct approximate stripped offset
            leading = len(m.group(0)) - len(m.group(0).lstrip())
            start = m.start() + leading
            stop = start + len(s)
            spans.append(TextSpan(s, start, stop))
        return spans


def iter_word_tokens(sentence_text: str) -> Iterable[TextSpan]:
    """Tokenize a sentence and yield word-like tokens; falls back if razdel is absent."""
    try:
        from razdel import tokenize  # type: ignore
        for t in tokenize(sentence_text):
            if re.search(r"[0-9A-Za-zА-Яа-яЁё]", t.text):
                yield TextSpan(t.text, t.start, t.stop)
        return
    except Exception:
        for m in re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+(?:-[0-9A-Za-zА-Яа-яЁё]+)?", sentence_text):
            yield TextSpan(m.group(0), m.start(), m.end())
