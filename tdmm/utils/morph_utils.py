from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):  # type: ignore
        fs = inspect.getfullargspec(func)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)

    inspect.getargspec = getargspec  # type: ignore

try:
    import pymorphy2  # type: ignore
except Exception:  # noqa: BLE001
    pymorphy2 = None  # type: ignore

_MORPH: Optional["pymorphy2.MorphAnalyzer"] = None

# Minimal fallback for environments where dependencies have not been installed yet.
# Full thesis runs should use pymorphy2; this table keeps smoke-tests meaningful.
_FALLBACK_LEMMAS = {
    "думал": "думать", "думала": "думать", "думали": "думать", "думаю": "думать", "думаешь": "думать", "думает": "думать", "думать": "думать",
    "подумал": "подумать", "подумала": "подумать", "подумали": "подумать", "подумав": "подумать", "подумаешь": "подумать", "подумать": "подумать",
    "вспомнил": "вспомнить", "вспомнила": "вспомнить", "вспомнили": "вспомнить", "вспомнить": "вспомнить",
    "припомнил": "припомнить", "припомнила": "припомнить", "припомнить": "припомнить",
    "помнил": "помнить", "помнила": "помнить", "помнить": "помнить",
    "восстановил": "восстановить", "восстановила": "восстановить", "восстановить": "восстановить",
    "стал": "стать", "стала": "стать", "стало": "стать", "стали": "стать", "стать": "стать",
    "сделалось": "сделаться", "сделался": "сделаться", "сделалась": "сделаться", "сделаться": "сделаться",
    "ясно": "ясно", "понятно": "понятно",
    "пойми": "понять", "понял": "понять", "поняла": "понять", "поняли": "понять", "понять": "понять",
    "решил": "решить", "решила": "решить", "решили": "решить", "решить": "решить", "решал": "решать", "решала": "решать", "решать": "решать",
    "мысленно": "мысленно", "проговорил": "проговорить", "проговорила": "проговорить", "проговорить": "проговорить",
    "произнес": "произнести", "произнёс": "произнести", "произнесла": "произнести", "сказал": "сказать", "сказала": "сказать",
    "пораскинь": "пораскинуть", "пораскинул": "пораскинуть", "пораскинуть": "пораскинуть",
    "мозгами": "мозг", "мозг": "мозг", "мозгов": "мозг", "голову": "голова", "голове": "голова", "головы": "голова", "голова": "голова",
    "памяти": "память", "память": "память", "ум": "ум", "уме": "ум", "ума": "ум", "умом": "ум",
    "мысль": "мысль", "мысли": "мысль", "мыслью": "мысль", "мыслей": "мысль",
    "сомнение": "сомнение", "сомнения": "сомнение", "сомнении": "сомнение",
    "разговор": "разговор", "детали": "деталь", "деталь": "деталь", "дело": "дело",
    "пришла": "прийти", "пришел": "прийти", "пришёл": "прийти", "пришло": "прийти", "прийти": "прийти",
    "мелькнула": "мелькнуть", "мелькнул": "мелькнуть", "мелькнуть": "мелькнуть",
    "возникла": "возникнуть", "возник": "возникнуть", "возникнуть": "возникнуть",
    "догадался": "догадаться", "догадалась": "догадаться", "догадаться": "догадаться",
}


def get_morph() -> "pymorphy2.MorphAnalyzer":
    global _MORPH
    if pymorphy2 is None:
        raise RuntimeError("NO_MORPH")
    if _MORPH is None:
        _MORPH = pymorphy2.MorphAnalyzer()
    return _MORPH


def _fallback_lemma(token: str) -> Tuple[str, str]:
    t = token.lower().replace("ё", "е").strip()
    if t in _FALLBACK_LEMMAS:
        return _FALLBACK_LEMMAS[t], "UNK"
    return t, "UNK"


@lru_cache(maxsize=200000)
def lemmatize_token(token: str) -> Tuple[str, str]:
    try:
        m = get_morph()
    except RuntimeError:
        return _fallback_lemma(token)
    parses = m.parse(token)
    if not parses:
        return token.lower(), "UNK"
    best = parses[0]
    return best.normal_form, best.tag.POS or "UNK"
