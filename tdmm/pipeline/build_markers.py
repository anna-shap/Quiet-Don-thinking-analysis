from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from tdmm.utils.morph_utils import lemmatize_token
from tdmm.utils.storage_utils import save_df


def _read_alex_list(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines: List[str] = []
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _theme_for_alex(unit: str) -> str:
    u = unit.lower()
    if any(x in u for x in ["вспомин", "припомин", "помн", "памят"]):
        return "память / воспоминание"
    if any(x in u for x in ["пон", "уразум", "догад", "осозна"]):
        return "понимание / ясность"
    if any(x in u for x in ["реш", "задум"]):
        return "решение / намерение"
    if any(x in u for x in ["сомнев", "колеб"]):
        return "сомнение / колебание"
    if any(x in u for x in ["мысленно"]):
        return "внутренняя речь"
    if any(x in u for x in ["мысл"]):
        return "инициация мысли"
    if any(x in u for x in ["дум", "размыш", "раздум", "обдум", "рассуж", "соображ"]):
        return "процесс размышления"
    return "лексико-тематическая группа мышления"


def _truthy(x: object) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "да", "y"}




def _load_microtheme_aliases(path: str | Path | None) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if not path:
        return aliases
    p = Path(path)
    if not p.exists():
        return aliases
    try:
        df = pd.read_csv(p, encoding="utf-8-sig").fillna("")
    except Exception:
        return aliases
    for _, r in df.iterrows():
        canon = str(r.get("canonical_microtheme", "")).strip()
        if not canon:
            continue
        variants = [canon] + [x.strip() for x in str(r.get("aliases", "")).split(";") if x.strip()]
        for v in variants:
            aliases[v.lower().replace("ё", "е").strip()] = canon
    return aliases


def _normalize_microtheme(value: object, aliases: Dict[str, str]) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("ё", "е")
    if key in aliases:
        return aliases[key]
    compact = key.replace(" / ", "/").replace("/", " / ")
    if compact in aliases:
        return aliases[compact]
    return raw or "лексико-тематическая группа мышления"

def _default_precision(marker_source: str, microtheme: str, unit_type: str) -> str:
    if marker_source in {"S1", "ALEXANDROVA"}:
        return "CORE"
    m = (microtheme or "").lower()
    if any(x in m for x in ["эмоционально", "навязчивая", "голова / ум", "интуитив"]):
        return "EXTENDED"
    return "CORE" if unit_type.upper() == "PHRASE" else "CORE"


def _normalize_optional_cols(df: pd.DataFrame, *, source: str, unit_type: str) -> pd.DataFrame:
    df = df.copy()
    if "precision_level" not in df.columns:
        df["precision_level"] = df.get("microtheme", "").apply(lambda m: _default_precision(source, str(m), unit_type)) if "microtheme" in df.columns else _default_precision(source, "", unit_type)
    df["precision_level"] = df["precision_level"].fillna("CORE").astype(str).str.upper().replace({"": "CORE"})
    if "source_type" not in df.columns:
        df["source_type"] = "curated" if source == "S1" else ("manual_phrase" if source == "PHRASE" else "dictionary")
    if "enabled" not in df.columns:
        df["enabled"] = 1
    if "risk_comment" not in df.columns:
        df["risk_comment"] = ""
    if "comment" not in df.columns:
        df["comment"] = ""
    df = df[df["enabled"].apply(_truthy)].copy()
    return df


def build_markers_master(
    *,
    s1_csv: str | Path,
    alexandrova_txt: str | Path,
    phrases_csv: str | Path,
    out_path: str | Path,
    export_csv: bool = False,
    csv_max_rows: int = 200_000,
    enabled_precision_levels: Optional[Sequence[str]] = None,
    microthemes_csv: str | Path | None = None,
) -> Path:
    """Build a unified MARKERS_MASTER with quality-control metadata.

    Main output columns: marker_id, marker_source, microtheme, unit_type,
    unit_raw, unit_norm, marker_label, precision_level, source_type, enabled,
    risk_comment, comment.
    """

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed_levels = {str(x).upper() for x in enabled_precision_levels} if enabled_precision_levels else None
    microtheme_aliases = _load_microtheme_aliases(microthemes_csv)

    # --- S1 (curated CSV) ---
    s1 = pd.read_csv(s1_csv, encoding="utf-8-sig").fillna("")
    s1 = _normalize_optional_cols(s1, source="S1", unit_type="WORD")
    s1["marker_source"] = "S1"
    s1["marker_id"] = s1["id"].astype(str)
    s1["unit_raw"] = s1["unit"].astype(str)
    s1["unit_type"] = s1["unit_type"].astype(str).str.upper()
    s1["microtheme"] = s1.get("microtheme", "").astype(str).apply(lambda v: _normalize_microtheme(v, microtheme_aliases))
    s1["phrase_family"] = "WORD_CORE"

    def _norm_s1(row: pd.Series) -> str:
        if str(row["unit_type"]).upper() == "WORD":
            return lemmatize_token(str(row["unit_raw"]))[0]
        return str(row["unit_raw"]).strip()

    s1["unit_norm"] = s1.apply(_norm_s1, axis=1)
    s1["marker_label"] = s1.apply(
        lambda r: (r["unit_norm"] if r["unit_type"] == "WORD" else r["unit_raw"])
        + (f" [{r['microtheme']}]" if str(r["microtheme"]).strip() else ""),
        axis=1,
    )

    cols = ["marker_id", "marker_source", "microtheme", "phrase_family", "unit_type", "unit_raw", "unit_norm", "marker_label", "precision_level", "source_type", "enabled", "risk_comment", "comment"]
    s1_out = s1[cols].copy()

    # --- Alexandrovа (single-column wordlist) ---
    alex_units = _read_alex_list(alexandrova_txt)
    a = pd.DataFrame({"unit_raw": alex_units})
    if len(a) > 0:
        a["marker_source"] = "ALEXANDROVA"
        a["unit_type"] = "WORD"
        a["unit_norm"] = a["unit_raw"].astype(str).apply(lambda s: lemmatize_token(s)[0])
        a["microtheme"] = a["unit_norm"].astype(str).apply(_theme_for_alex).apply(lambda v: _normalize_microtheme(v, microtheme_aliases))
        a["phrase_family"] = "WORD_ALEXANDROVA"
        a["marker_id"] = [f"ALEX_{i:03d}" for i in range(1, len(a) + 1)]
        a["marker_label"] = a["unit_norm"].astype(str) + " [" + a["microtheme"].astype(str) + "]"
        a["precision_level"] = "CORE"
        a["source_type"] = "dictionary"
        a["enabled"] = 1
        a["risk_comment"] = ""
        a["comment"] = "дополнительная словарная единица"
        a_out = a[cols].copy()
    else:
        a_out = pd.DataFrame(columns=cols)

    # --- Phrases ---
    phr = pd.read_csv(phrases_csv, encoding="utf-8-sig").fillna("")
    phr = _normalize_optional_cols(phr, source="PHRASE", unit_type="PHRASE")
    phr["marker_source"] = "PHRASE"
    phr["unit_type"] = "PHRASE"

    def _unit_norm_phrase(r: pd.Series) -> str:
        pat = str(r.get("pattern_lemmas", "") or "").strip()
        if pat:
            return pat
        return str(r.get("phrase_text", "")).strip()

    phr["unit_norm"] = phr.apply(_unit_norm_phrase, axis=1)
    phr["unit_raw"] = phr.get("phrase_text", "").astype(str)
    phr["microtheme"] = phr.get("microtheme", "").astype(str).apply(lambda v: _normalize_microtheme(v, microtheme_aliases))
    if "phrase_family" not in phr.columns:
        phr["phrase_family"] = ""
    phr["phrase_family"] = phr["phrase_family"].astype(str).replace({"": "PHRASE_GENERAL"})
    phr["marker_id"] = phr.get("id", pd.Series(range(1, len(phr) + 1))).astype(str)
    phr["marker_id"] = phr["marker_id"].apply(lambda x: x if str(x).startswith("PHR_") else f"PHR_{int(x):03d}" if str(x).isdigit() else str(x))
    phr["marker_label"] = phr["unit_raw"].astype(str) + " [" + phr["microtheme"].astype(str) + "]"
    phr_out = phr[cols].copy()

    # --- merge, filter by precision, and deduplicate ---
    allm = pd.concat([phr_out, s1_out, a_out], ignore_index=True)
    allm["unit_norm"] = allm["unit_norm"].astype(str).str.strip()
    allm["unit_type"] = allm["unit_type"].astype(str).str.upper()
    allm["precision_level"] = allm["precision_level"].astype(str).str.upper().replace({"": "CORE"})

    if allowed_levels is not None:
        allm = allm[allm["precision_level"].isin(allowed_levels)].copy()

    # priority: phrase patterns first, then curated S1, then Alexandrovа
    priority = {"PHRASE": 0, "S1": 1, "ALEXANDROVA": 2}
    allm["_prio"] = allm["marker_source"].map(priority).fillna(99).astype(int)
    allm = allm.sort_values(["unit_type", "unit_norm", "_prio"]).drop_duplicates(["unit_type", "unit_norm"], keep="first")
    allm = allm.drop(columns=["_prio"]).reset_index(drop=True)

    allm = allm.sort_values(["marker_source", "precision_level", "marker_id"]).reset_index(drop=True)
    return save_df(allm, out, export_csv=export_csv, csv_max_rows=csv_max_rows)
