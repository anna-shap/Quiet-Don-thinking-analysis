from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from tdmm.utils.storage_utils import load_df


_CONF_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _linked_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    mode = (mode or "WIDE").upper()
    if mode == "STRICT":
        return df["MelekhovLink"].astype(str) == "YES"
    return df["MelekhovLink"].astype(str).isin(["YES", "PROBABLE"])


def _pct(count: pd.Series, total: int) -> pd.Series:
    if total <= 0:
        return pd.Series([0.0] * len(count), index=count.index)
    return (count.astype(float) / float(total) * 100.0).round(2)


def _group_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count", "share_pct"])
    out = df.groupby(col, dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    out["share_pct"] = _pct(out["count"], len(df))
    return out


def _merge_marker_meta(hits: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "marker_id", "marker_label", "microtheme", "unit_type", "marker_source",
        "precision_level", "source_type", "risk_comment", "unit_raw", "unit_norm", "phrase_family",
    ]
    cols = [c for c in meta_cols if c in markers.columns]
    out = hits.copy()
    out["marker_id"] = out["marker_id"].astype(str)
    markers = markers.copy()
    markers["marker_id"] = markers["marker_id"].astype(str)
    existing_meta = [c for c in cols if c != "marker_id" and c in out.columns]
    if existing_meta:
        out = out.drop(columns=existing_meta)
    return out.merge(markers[cols], on="marker_id", how="left")


def _filter_counted(
    hits_raw: pd.DataFrame,
    *,
    include_noise_in_counts: bool,
    enabled_precision_levels: Sequence[str] | None,
    experimental_phrases_in_main_stats: bool,
    min_perspective_confidence_main: str,
) -> pd.DataFrame:
    hits = hits_raw.copy()
    if not include_noise_in_counts and "is_noise" in hits.columns:
        hits = hits[hits["is_noise"] != True].copy()  # noqa: E712
    if "count_in_main_stats" in hits.columns:
        hits = hits[hits["count_in_main_stats"] != False].copy()  # noqa: E712
    if enabled_precision_levels:
        allowed = {str(x).upper() for x in enabled_precision_levels}
        hits = hits[hits.get("precision_level", "CORE").astype(str).str.upper().isin(allowed)].copy()
    if not experimental_phrases_in_main_stats:
        hits = hits[hits.get("precision_level", "CORE").astype(str).str.upper() != "EXPERIMENTAL"].copy()
    min_rank = _CONF_RANK.get(str(min_perspective_confidence_main).upper(), 1)
    if "PerspectiveConfidence" in hits.columns:
        ranks = hits["PerspectiveConfidence"].astype(str).str.upper().map(_CONF_RANK).fillna(1).astype(int)
        hits = hits[ranks >= min_rank].copy()
    return hits


def _by_marker(df: pd.DataFrame, total_sents: int, total_words: int, total_ref: int | None = None) -> pd.DataFrame:
    cols = ["marker_id", "count", "share_pct", "per_1000_sentences", "per_10000_words", "marker_label", "unit_type", "microtheme", "phrase_family", "marker_source", "precision_level"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "phrase_family" not in df.columns:
        df = df.copy(); df["phrase_family"] = ""
    c = df.groupby(["marker_id", "marker_label", "unit_type", "microtheme", "phrase_family", "marker_source", "precision_level"], dropna=False).size().reset_index(name="count")
    denom = int(total_ref) if total_ref is not None else int(len(df))
    c["share_pct"] = _pct(c["count"], denom)
    c["per_1000_sentences"] = (c["count"] / total_sents * 1000).round(3) if total_sents else 0.0
    c["per_10000_words"] = (c["count"] / total_words * 10000).round(3) if total_words else 0.0
    return c.sort_values("count", ascending=False)[cols]


def _segment_normalized(df: pd.DataFrame, sents: pd.DataFrame, seg_csv: str | Path | None, *, seg_name: str, out_col: str) -> pd.DataFrame:
    if df.empty or not seg_csv or not Path(seg_csv).exists() or out_col not in df.columns:
        return pd.DataFrame(columns=[out_col, "sentences", "words", "hits", "hits_per_1000_sentences", "hits_per_10000_words"])
    seg = pd.read_csv(seg_csv)
    if "start_sent" not in seg.columns or "end_sent" not in seg.columns:
        return pd.DataFrame()
    name_col = f"{seg_name}_name"
    id_col = f"{seg_name}_id"
    if name_col not in seg.columns:
        # tolerate old simple files
        alt = "part_name" if seg_name == "part" else "book_name"
        name_col = alt if alt in seg.columns else seg.columns[0]
    rows = []
    for _, r in seg.iterrows():
        start, end = int(r["start_sent"]), int(r["end_sent"])
        ss = sents[(sents["sent_id"].astype(int) >= start) & (sents["sent_id"].astype(int) <= end)]
        hits_n = int(((df["sent_id"].astype(int) >= start) & (df["sent_id"].astype(int) <= end)).sum())
        sent_n = int(len(ss))
        word_n = int(ss.get("word_count", pd.Series(dtype=int)).sum()) if "word_count" in ss.columns else 0
        rows.append({
            out_col: str(r.get(name_col, r.get(id_col, "UNKNOWN"))),
            "start_sent": start,
            "end_sent": end,
            "sentences": sent_n,
            "words": word_n,
            "hits": hits_n,
            "hits_per_1000_sentences": round(hits_n / sent_n * 1000, 3) if sent_n else 0.0,
            "hits_per_10000_words": round(hits_n / word_n * 10000, 3) if word_n else 0.0,
        })
    return pd.DataFrame(rows)


def build_phrase_quality_report(out_dir: str | Path, hits_raw: pd.DataFrame, markers: pd.DataFrame) -> Path:
    out_dir = Path(out_dir)
    phrases = markers[markers["unit_type"].astype(str).str.upper() == "PHRASE"].copy()
    phrase_hits = hits_raw[hits_raw["hit_type"].astype(str).str.upper() == "PHRASE"].copy() if "hit_type" in hits_raw.columns else pd.DataFrame()

    phrase_coverage = phrases[[c for c in ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_raw", "unit_norm", "precision_level", "source_type", "risk_comment"] if c in phrases.columns]].copy()
    counts = phrase_hits.groupby("marker_id").size().reset_index(name="count_all") if not phrase_hits.empty else pd.DataFrame(columns=["marker_id", "count_all"])
    phrase_coverage = phrase_coverage.merge(counts, on="marker_id", how="left")
    phrase_coverage["count_all"] = phrase_coverage["count_all"].fillna(0).astype(int)
    phrase_coverage["appears_in_corpus"] = phrase_coverage["count_all"] > 0
    phrase_coverage = phrase_coverage.sort_values(["count_all", "precision_level", "marker_id"], ascending=[False, True, True])

    phrase_zero = phrase_coverage[phrase_coverage["count_all"] == 0].copy()
    phrase_top = phrase_coverage[phrase_coverage["count_all"] > 0].head(100).copy()

    overlap_rows = pd.DataFrame()
    if "overlaps_phrase" in hits_raw.columns:
        word_over = hits_raw[(hits_raw["hit_type"].astype(str).str.upper() == "WORD") & (hits_raw["overlaps_phrase"] == True)].copy()  # noqa: E712
        if not word_over.empty:
            overlap_rows = word_over.groupby(["phrase_overlap_ids", "target_lemma", "target_surface"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)

    examples = pd.DataFrame()
    if not phrase_hits.empty:
        ex_cols = [c for c in ["hit_id", "marker_id", "marker_label", "microtheme", "precision_level", "target_surface", "sent_text", "context_window", "MelekhovLink", "Perspective", "PerspectiveEvidence"] if c in phrase_hits.columns]
        examples = phrase_hits.sort_values(["marker_id", "sent_id"]).groupby("marker_id", as_index=False).head(3)[ex_cols]

    by_precision = _group_count(phrases, "precision_level")
    by_family = _group_count(phrase_hits.merge(phrases[["marker_id", "phrase_family"]], on="marker_id", how="left") if (not phrase_hits.empty and "phrase_family" in phrases.columns) else pd.DataFrame(), "phrase_family")
    try:
        nested_candidates = load_df(out_dir / "PHRASE_NESTED_CANDIDATES")
    except Exception:
        nested_candidates = pd.DataFrame()

    manual = examples.copy()
    if not manual.empty:
        manual["human_valid"] = ""
        manual["human_comment"] = ""

    out_xlsx = out_dir / "PHRASE_QUALITY_REPORT.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        phrase_coverage.to_excel(xw, sheet_name="phrase_coverage", index=False)
        phrase_zero.to_excel(xw, sheet_name="phrase_zero_hits", index=False)
        phrase_top.to_excel(xw, sheet_name="phrase_top_hits", index=False)
        overlap_rows.to_excel(xw, sheet_name="phrase_overlaps", index=False)
        examples.to_excel(xw, sheet_name="phrase_examples", index=False)
        by_precision.to_excel(xw, sheet_name="phrase_by_precision", index=False)
        by_family.to_excel(xw, sheet_name="phrase_by_family", index=False)
        nested_candidates.to_excel(xw, sheet_name="nested_candidates", index=False)
        manual.to_excel(xw, sheet_name="manual_phrase_check", index=False)
    return out_xlsx



def _pivot_matrix(df: pd.DataFrame, index_col: str, column_col: str) -> pd.DataFrame:
    if df.empty or index_col not in df.columns or column_col not in df.columns:
        return pd.DataFrame()
    mat = df.groupby([index_col, column_col], dropna=False).size().reset_index(name="count")
    return mat.pivot(index=index_col, columns=column_col, values="count").fillna(0).astype(int).reset_index()


def _part_indicator(df: pd.DataFrame, perspective_values: Sequence[str], out_name: str) -> pd.DataFrame:
    if df.empty or "Part" not in df.columns or "Perspective" not in df.columns:
        return pd.DataFrame(columns=["Part", "all_hits", out_name, "share_pct"])
    all_counts = df.groupby("Part", dropna=False).size().reset_index(name="all_hits")
    sub = df[df["Perspective"].astype(str).isin(list(perspective_values))].groupby("Part", dropna=False).size().reset_index(name=out_name)
    out = all_counts.merge(sub, on="Part", how="left")
    out[out_name] = out[out_name].fillna(0).astype(int)
    out["share_pct"] = (out[out_name] / out["all_hits"] * 100).round(2).fillna(0.0)
    return out.sort_values("Part")


def _hero_vs_corpus_microthemes(counted: pd.DataFrame, strict: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in [("CORPUS_COUNTED", counted), ("MELEKHOV_STRICT", strict), ("MELEKHOV_WIDE", wide)]:
        if frame.empty or "microtheme" not in frame.columns:
            continue
        g = frame.groupby("microtheme", dropna=False).size().reset_index(name="count")
        total = int(g["count"].sum())
        g["share_pct"] = (g["count"] / total * 100).round(2) if total else 0.0
        g["scope"] = scope
        rows.append(g[["scope", "microtheme", "count", "share_pct"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["scope", "microtheme", "count", "share_pct"])




def _analytical_indices(linked: pd.DataFrame, part_normalized: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable thesis-level indices by Part.

    The values are ratios, not claims of psychological measurement. They are
    useful for comparing textual segments of different size.
    """
    cols = [
        "Part", "all_hits", "cognitive_density_per_10000_words",
        "inner_speech_hits", "inner_speech_index_pct",
        "external_pressure_hits", "external_pressure_index_pct",
        "core_hits", "extended_hits", "extended_cognition_index_pct",
    ]
    if linked.empty or "Part" not in linked.columns:
        return pd.DataFrame(columns=cols)
    base = linked.groupby("Part", dropna=False).size().reset_index(name="all_hits")
    inner = linked[linked.get("Perspective", "").astype(str) == "HERO_INNER_SPEECH"].groupby("Part", dropna=False).size().reset_index(name="inner_speech_hits") if "Perspective" in linked.columns else pd.DataFrame(columns=["Part", "inner_speech_hits"])
    external = linked[linked.get("Perspective", "").astype(str).isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"])].groupby("Part", dropna=False).size().reset_index(name="external_pressure_hits") if "Perspective" in linked.columns else pd.DataFrame(columns=["Part", "external_pressure_hits"])
    core = linked[linked.get("precision_level", "").astype(str).str.upper() == "CORE"].groupby("Part", dropna=False).size().reset_index(name="core_hits") if "precision_level" in linked.columns else pd.DataFrame(columns=["Part", "core_hits"])
    ext = linked[linked.get("precision_level", "").astype(str).str.upper() == "EXTENDED"].groupby("Part", dropna=False).size().reset_index(name="extended_hits") if "precision_level" in linked.columns else pd.DataFrame(columns=["Part", "extended_hits"])
    out = base.merge(inner, on="Part", how="left").merge(external, on="Part", how="left").merge(core, on="Part", how="left").merge(ext, on="Part", how="left").fillna(0)
    for c in ["inner_speech_hits", "external_pressure_hits", "core_hits", "extended_hits"]:
        out[c] = out[c].astype(int)
    out["inner_speech_index_pct"] = (out["inner_speech_hits"] / out["all_hits"] * 100).round(2).fillna(0.0)
    out["external_pressure_index_pct"] = (out["external_pressure_hits"] / out["all_hits"] * 100).round(2).fillna(0.0)
    denom = (out["core_hits"] + out["extended_hits"]).replace(0, pd.NA)
    out["extended_cognition_index_pct"] = (out["extended_hits"] / denom * 100).round(2).fillna(0.0)
    if not part_normalized.empty and "Part" in part_normalized.columns and "hits_per_10000_words" in part_normalized.columns:
        out = out.merge(part_normalized[["Part", "hits_per_10000_words"]].rename(columns={"hits_per_10000_words": "cognitive_density_per_10000_words"}), on="Part", how="left")
    else:
        out["cognitive_density_per_10000_words"] = 0.0
    return out[cols]


def _melekhov_cognitive_profile(linked: pd.DataFrame) -> pd.DataFrame:
    if linked.empty or "microtheme" not in linked.columns:
        return pd.DataFrame(columns=["microtheme", "count", "share_pct", "core_hits", "extended_hits", "inner_speech_hits", "narrator_hits", "external_hits"])
    base = linked.groupby("microtheme", dropna=False).size().reset_index(name="count")
    total = int(base["count"].sum())
    base["share_pct"] = (base["count"] / total * 100).round(2) if total else 0.0
    def g(mask, name):
        sub = linked[mask].groupby("microtheme", dropna=False).size().reset_index(name=name) if not linked.empty else pd.DataFrame(columns=["microtheme", name])
        return sub
    core = g(linked.get("precision_level", "").astype(str).str.upper() == "CORE", "core_hits") if "precision_level" in linked.columns else pd.DataFrame(columns=["microtheme", "core_hits"])
    ext = g(linked.get("precision_level", "").astype(str).str.upper() == "EXTENDED", "extended_hits") if "precision_level" in linked.columns else pd.DataFrame(columns=["microtheme", "extended_hits"])
    inner = g(linked.get("Perspective", "").astype(str) == "HERO_INNER_SPEECH", "inner_speech_hits") if "Perspective" in linked.columns else pd.DataFrame(columns=["microtheme", "inner_speech_hits"])
    narr = g(linked.get("Perspective", "").astype(str) == "NARRATOR_ABOUT_HERO", "narrator_hits") if "Perspective" in linked.columns else pd.DataFrame(columns=["microtheme", "narrator_hits"])
    external = g(linked.get("Perspective", "").astype(str).isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"]), "external_hits") if "Perspective" in linked.columns else pd.DataFrame(columns=["microtheme", "external_hits"])
    out = base.merge(core, on="microtheme", how="left").merge(ext, on="microtheme", how="left").merge(inner, on="microtheme", how="left").merge(narr, on="microtheme", how="left").merge(external, on="microtheme", how="left").fillna(0)
    for c in ["core_hits", "extended_hits", "inner_speech_hits", "narrator_hits", "external_hits"]:
        out[c] = out[c].astype(int)
    return out.sort_values("count", ascending=False)


def build_manual_verification(out_dir: str | Path, *, hits_raw: pd.DataFrame, counted: pd.DataFrame) -> Path:
    out_dir = Path(out_dir)
    def _prep(df: pd.DataFrame, n: int = 250) -> pd.DataFrame:
        cols = [c for c in ["hit_id", "marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level", "target_surface", "sent_text", "context_window", "MelekhovLink", "MelekhovEvidence", "Perspective", "PerspectiveConfidence", "PerspectiveEvidence", "SpeakerDetected", "AddresseeDetected", "OthersSpeechAct", "Part", "Book", "is_noise", "noise_reason"] if c in df.columns]
        out = df[cols].head(n).copy() if not df.empty else pd.DataFrame(columns=cols)
        for c in ["human_valid", "human_melekhov_link", "human_perspective", "human_comment"]:
            out[c] = ""
        return out

    hit_check = _prep(counted.sort_values(["MelekhovLink", "sent_id", "token_idx"]) if {"MelekhovLink", "sent_id", "token_idx"}.issubset(counted.columns) else counted, 300)
    phrase_check = _prep(counted[counted.get("hit_type", "").astype(str).str.upper() == "PHRASE"].sort_values(["precision_level", "marker_id"]) if "hit_type" in counted.columns else pd.DataFrame(), 300)
    perspective_check = _prep(counted[counted.get("PerspectiveConfidence", "").astype(str).str.upper().isin(["LOW", "MEDIUM"])].copy() if "PerspectiveConfidence" in counted.columns else pd.DataFrame(), 300)
    probable_check = _prep(counted[counted.get("MelekhovLink", "").astype(str) == "PROBABLE"].copy() if "MelekhovLink" in counted.columns else pd.DataFrame(), 300)
    noise_check = _prep(hits_raw[hits_raw.get("is_noise", False) == True].copy() if "is_noise" in hits_raw.columns else pd.DataFrame(), 300)  # noqa: E712
    summary = pd.DataFrame([
        {"sheet": "hit_check", "purpose": "общая выборка засчитанных попаданий"},
        {"sheet": "phrase_check", "purpose": "ручная проверка фразовых маркеров"},
        {"sheet": "perspective_check", "purpose": "проверка речевой перспективы с низкой/средней уверенностью"},
        {"sheet": "probable_link_check", "purpose": "проверка вероятной связи с Григорием Мелеховым"},
        {"sheet": "noise_check", "purpose": "проверка исключённых шумовых употреблений"},
    ])
    out_xlsx = out_dir / "MANUAL_VERIFICATION.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="README", index=False)
        hit_check.to_excel(xw, sheet_name="hit_check", index=False)
        phrase_check.to_excel(xw, sheet_name="phrase_check", index=False)
        perspective_check.to_excel(xw, sheet_name="perspective_check", index=False)
        probable_check.to_excel(xw, sheet_name="probable_link_check", index=False)
        noise_check.to_excel(xw, sheet_name="noise_check", index=False)
    return out_xlsx

def build_quality_dashboard(
    out_dir: str | Path,
    *,
    hits_raw: pd.DataFrame,
    counted: pd.DataFrame,
    markers: pd.DataFrame,
    noise_report: pd.DataFrame,
    nested_phrase_candidates: pd.DataFrame,
) -> Path:
    out_dir = Path(out_dir)
    warnings = []
    if "microtheme" in markers.columns and markers["microtheme"].astype(str).str.strip().eq("").any():
        warnings.append({"level": "WARN", "check": "empty_microtheme", "details": "Есть маркеры с пустой микротемой"})
    if "marker_id" in markers.columns and markers["marker_id"].duplicated().any():
        warnings.append({"level": "ERROR", "check": "duplicate_marker_id", "details": "Есть дубли marker_id"})
    if not nested_phrase_candidates.empty:
        warnings.append({"level": "INFO", "check": "nested_phrase_candidates", "details": f"Вложенных фраз-кандидатов: {len(nested_phrase_candidates)}"})
    if "Perspective" in counted.columns:
        unk = int((counted["Perspective"].astype(str) == "UNKNOWN").sum())
        if unk:
            warnings.append({"level": "WARN", "check": "unknown_perspective", "details": f"Неопределённая перспектива: {unk}"})
    if "PerspectiveConfidence" in counted.columns:
        low = int((counted["PerspectiveConfidence"].astype(str).str.upper() == "LOW").sum())
        if low:
            warnings.append({"level": "INFO", "check": "low_confidence", "details": f"Низкая уверенность: {low}"})

    summary = pd.DataFrame([
        {"metric": "markers_total", "value": int(len(markers))},
        {"metric": "phrase_markers", "value": int((markers.get("unit_type", "").astype(str).str.upper() == "PHRASE").sum()) if "unit_type" in markers.columns else 0},
        {"metric": "hits_raw", "value": int(len(hits_raw))},
        {"metric": "hits_counted", "value": int(len(counted))},
        {"metric": "noise_hits", "value": int((hits_raw.get("is_noise", False) == True).sum()) if "is_noise" in hits_raw.columns else 0},
        {"metric": "nested_phrase_candidates", "value": int(len(nested_phrase_candidates))},
    ])

    exact_dups = markers[markers["marker_id"].duplicated(keep=False)].sort_values("marker_id") if "marker_id" in markers.columns else pd.DataFrame()
    lemma_dups = markers[markers.duplicated(["unit_type", "unit_norm"], keep=False)].sort_values(["unit_type", "unit_norm"]) if {"unit_type", "unit_norm"}.issubset(markers.columns) else pd.DataFrame()
    low_conf = counted[counted.get("PerspectiveConfidence", "").astype(str).str.upper() == "LOW"].copy() if "PerspectiveConfidence" in counted.columns else pd.DataFrame()
    unknown = counted[counted.get("Perspective", "").astype(str) == "UNKNOWN"].copy() if "Perspective" in counted.columns else pd.DataFrame()
    manual_required = pd.concat([
        unknown.head(100),
        low_conf.head(100),
        counted[counted.get("MelekhovLink", "").astype(str) == "PROBABLE"].head(100) if "MelekhovLink" in counted.columns else pd.DataFrame(),
    ], ignore_index=True).drop_duplicates(subset=["hit_id"] if "hit_id" in counted.columns else None).head(300)

    out_xlsx = out_dir / "QUALITY_DASHBOARD.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(warnings, columns=["level", "check", "details"]).to_excel(xw, sheet_name="warnings", index=False)
        exact_dups.to_excel(xw, sheet_name="duplicate_ids", index=False)
        lemma_dups.to_excel(xw, sheet_name="duplicate_lemmas", index=False)
        nested_phrase_candidates.head(5000).to_excel(xw, sheet_name="nested_phrases", index=False)
        noise_report.to_excel(xw, sheet_name="noise", index=False)
        low_conf.head(1000).to_excel(xw, sheet_name="low_confidence", index=False)
        unknown.head(1000).to_excel(xw, sheet_name="unknown_perspective", index=False)
        manual_required.to_excel(xw, sheet_name="manual_required", index=False)
    return out_xlsx

def make_stats(
    *,
    out_dir: str | Path,
    segmentation_parts_csv: str | Path | None = None,
    segmentation_books_csv: str | Path | None = None,
    include_noise_in_counts: bool = False,
    melekhov_link_mode: str = "WIDE",
    analysis_profile: str = "DIPLOMA_STRICT",
    count_overlapping_word_hits: bool = False,
    enabled_precision_levels: Sequence[str] | None = None,
    experimental_phrases_in_main_stats: bool = False,
    save_phrase_quality_report: bool = True,
    min_perspective_confidence_main: str = "LOW",
) -> Path:
    out_dir = Path(out_dir)

    hits_loaded = load_df(out_dir / "TD_HITS_TAGGED").copy()
    markers = load_df(out_dir / "MARKERS_MASTER").copy()
    sents = load_df(out_dir / "SENTS").copy()
    sents["sent_id"] = sents["sent_id"].astype(int)

    hits_raw = _merge_marker_meta(hits_loaded, markers)
    if not count_overlapping_word_hits and "overlaps_phrase" in hits_raw.columns:
        # Preserve raw hits, but make the accounting decision explicit.
        mask = (hits_raw["hit_type"].astype(str).str.upper() == "WORD") & (hits_raw["overlaps_phrase"] == True)  # noqa: E712
        hits_raw.loc[mask, "count_in_main_stats"] = False

    counted = _filter_counted(
        hits_raw,
        include_noise_in_counts=include_noise_in_counts,
        enabled_precision_levels=enabled_precision_levels,
        experimental_phrases_in_main_stats=experimental_phrases_in_main_stats,
        min_perspective_confidence_main=min_perspective_confidence_main,
    )

    strict = counted[counted["MelekhovLink"].astype(str) == "YES"].copy()
    wide = counted[counted["MelekhovLink"].astype(str).isin(["YES", "PROBABLE"])].copy()
    linked = counted[_linked_mask(counted, melekhov_link_mode)].copy()

    total_words = int(sents.get("word_count", pd.Series(dtype=int)).sum()) if "word_count" in sents.columns else 0
    total_sents = int(len(sents))

    by_marker_all = _by_marker(counted, total_sents, total_words)
    by_marker_strict = _by_marker(strict, total_sents, total_words)
    by_marker_wide = _by_marker(wide, total_sents, total_words)
    by_marker_linked = _by_marker(linked, total_sents, total_words)

    cov_cols = ["marker_id", "marker_source", "microtheme", "phrase_family", "unit_type", "unit_raw", "unit_norm", "marker_label", "precision_level", "source_type", "risk_comment"]
    cov = markers[[c for c in cov_cols if c in markers.columns]].copy()
    for src, name in [(by_marker_all, "count_all"), (by_marker_strict, "count_strict"), (by_marker_wide, "count_wide")]:
        cov = cov.merge(src[["marker_id", "count"]].rename(columns={"count": name}), on="marker_id", how="left")
        cov[name] = cov[name].fillna(0).astype(int)
    cov["appears_in_corpus"] = cov["count_all"] > 0
    cov["appears_in_strict_hero"] = cov["count_strict"] > 0
    cov = cov.sort_values(["count_strict", "count_wide", "count_all"], ascending=False)

    by_link = _group_count(counted, "MelekhovLink")
    perspective_linked = _group_count(linked, "Perspective")
    perspective_strict = _group_count(strict, "Perspective")
    perspective_conf = _group_count(linked, "PerspectiveConfidence")
    by_microtheme_all = _group_count(counted, "microtheme")
    by_microtheme_linked = _group_count(linked, "microtheme")
    by_precision = _group_count(counted, "precision_level")
    by_others_speechact = _group_count(linked[linked.get("Perspective", "").astype(str).isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"])].copy(), "OthersSpeechAct") if "Perspective" in linked.columns else pd.DataFrame()

    marker_core = _by_marker(counted[counted["precision_level"].astype(str).str.upper() == "CORE"], total_sents, total_words)
    marker_extended = _by_marker(counted[counted["precision_level"].astype(str).str.upper() == "EXTENDED"], total_sents, total_words)
    marker_experimental = _by_marker(hits_raw[hits_raw.get("precision_level", "").astype(str).str.upper() == "EXPERIMENTAL"], total_sents, total_words)

    part_normalized = _segment_normalized(linked, sents, segmentation_parts_csv, seg_name="part", out_col="Part")
    book_normalized = _segment_normalized(linked, sents, segmentation_books_csv, seg_name="book", out_col="Book")

    perspective_microtheme_matrix = _pivot_matrix(linked, "Perspective", "microtheme")
    part_microtheme_matrix = _pivot_matrix(linked, "Part", "microtheme")
    core_extended_by_part = _pivot_matrix(linked, "Part", "precision_level")
    inner_speech_by_part = _part_indicator(linked, ["HERO_INNER_SPEECH"], "inner_speech_hits")
    external_pressure_by_part = _part_indicator(linked, ["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"], "external_pressure_hits")
    hero_vs_corpus_microthemes = _hero_vs_corpus_microthemes(counted, strict, wide)
    phrase_family_coverage = _group_count(linked[linked.get("unit_type", "").astype(str).str.upper() == "PHRASE"], "phrase_family")
    analytical_indices = _analytical_indices(linked, part_normalized)
    melekhov_cognitive_profile = _melekhov_cognitive_profile(linked)
    try:
        nested_phrase_candidates = load_df(out_dir / "PHRASE_NESTED_CANDIDATES")
    except Exception:
        nested_phrase_candidates = pd.DataFrame()

    by_part_top = pd.DataFrame()
    by_book_top = pd.DataFrame()
    if "Part" in linked.columns and not by_marker_linked.empty:
        top_ids = by_marker_linked.head(8)["marker_id"].astype(str).tolist()
        dyn = linked[linked["marker_id"].isin(top_ids)].copy()
        if len(dyn) > 0:
            by_part_top = dyn.groupby(["Part", "marker_id", "marker_label"], dropna=False).size().reset_index(name="count").sort_values(["Part", "count"], ascending=[True, False])
    if "Book" in linked.columns and not by_marker_linked.empty:
        top_ids = by_marker_linked.head(8)["marker_id"].astype(str).tolist()
        dyn = linked[linked["marker_id"].isin(top_ids)].copy()
        if len(dyn) > 0:
            by_book_top = dyn.groupby(["Book", "marker_id", "marker_label"], dropna=False).size().reset_index(name="count").sort_values(["Book", "count"], ascending=[True, False])

    noise_report = pd.DataFrame(columns=["noise_reason", "target_surface", "target_lemma", "count"])
    if "is_noise" in hits_raw.columns:
        noise = hits_raw[hits_raw["is_noise"] == True].copy()  # noqa: E712
        if len(noise) > 0:
            noise_report = noise.groupby(["noise_reason", "target_surface", "target_lemma"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)

    overlap_report = pd.DataFrame()
    if "overlaps_phrase" in hits_raw.columns:
        overlap_report = hits_raw[(hits_raw["hit_type"].astype(str).str.upper() == "WORD") & (hits_raw["overlaps_phrase"] == True)].copy()  # noqa: E712
        if not overlap_report.empty:
            overlap_report = overlap_report.groupby(["target_lemma", "target_surface", "phrase_overlap_ids"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)

    strict_vs_wide = pd.DataFrame([
        {"mode": "STRICT", "definition": "только MelekhovLink = YES", "count": int(len(strict))},
        {"mode": "WIDE", "definition": "MelekhovLink = YES + PROBABLE", "count": int(len(wide))},
        {"mode": "CONFIG_USED", "definition": str(melekhov_link_mode).upper(), "count": int(len(linked))},
    ])

    profile_settings = pd.DataFrame([
        {"setting": "analysis_profile", "value": analysis_profile},
        {"setting": "melekhov_link_mode", "value": str(melekhov_link_mode).upper()},
        {"setting": "include_noise_in_counts", "value": include_noise_in_counts},
        {"setting": "count_overlapping_word_hits", "value": count_overlapping_word_hits},
        {"setting": "enabled_precision_levels", "value": ", ".join(enabled_precision_levels or [])},
        {"setting": "experimental_phrases_in_main_stats", "value": experimental_phrases_in_main_stats},
        {"setting": "min_perspective_confidence_main", "value": min_perspective_confidence_main},
    ])

    overview = pd.DataFrame({
        "metric": [
            "sentences_total", "words_total", "hits_total_raw", "hits_total_counted",
            "hits_linked_wide", "hits_linked_strict", "noise_hits_excluded",
            "overlapping_word_hits_excluded", "unique_markers_in_registry", "phrase_markers_in_registry",
            "core_markers_in_registry", "extended_markers_in_registry", "experimental_markers_in_registry",
            "unique_markers_in_corpus_counted", "unique_markers_strict_hero", "unique_markers_wide_hero",
            "melekhov_link_mode_used", "manual_verification_file",
        ],
        "value": [
            total_sents, total_words, int(len(hits_raw)), int(len(counted)),
            int(len(wide)), int(len(strict)), int((hits_raw.get("is_noise", False) == True).sum()) if "is_noise" in hits_raw.columns else 0,  # noqa: E712
            int(((hits_raw.get("hit_type", "") == "WORD") & (hits_raw.get("overlaps_phrase", False) == True) & (hits_raw.get("count_in_main_stats", True) == False)).sum()) if "overlaps_phrase" in hits_raw.columns else 0,  # noqa: E712
            int(len(markers)), int((markers["unit_type"].astype(str).str.upper() == "PHRASE").sum()),
            int((markers.get("precision_level", "").astype(str).str.upper() == "CORE").sum()),
            int((markers.get("precision_level", "").astype(str).str.upper() == "EXTENDED").sum()),
            int((markers.get("precision_level", "").astype(str).str.upper() == "EXPERIMENTAL").sum()),
            int(by_marker_all["marker_id"].nunique()) if not by_marker_all.empty else 0,
            int(by_marker_strict["marker_id"].nunique()) if not by_marker_strict.empty else 0,
            int(by_marker_wide["marker_id"].nunique()) if not by_marker_wide.empty else 0,
            str(melekhov_link_mode).upper(), "MANUAL_VERIFICATION.xlsx",
        ],
    })

    manual_check_sample = hits_raw.copy()
    if len(manual_check_sample) > 0:
        priority = pd.Categorical(manual_check_sample.get("MelekhovLink", ""), categories=["YES", "PROBABLE", "NO"], ordered=True)
        manual_check_sample = manual_check_sample.assign(_priority=priority).sort_values(["_priority", "sent_id", "token_idx"]).drop(columns=["_priority"]).head(300)
        for col in ["human_valid", "human_perspective", "human_comment"]:
            manual_check_sample[col] = ""

    out_xlsx = out_dir / "STATS.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        overview.to_excel(xw, sheet_name="overview", index=False)
        profile_settings.to_excel(xw, sheet_name="profile_settings", index=False)
        cov.to_excel(xw, sheet_name="coverage", index=False)
        strict_vs_wide.to_excel(xw, sheet_name="strict_vs_wide", index=False)
        by_link.to_excel(xw, sheet_name="by_link", index=False)
        perspective_linked.to_excel(xw, sheet_name="perspective_linked", index=False)
        perspective_strict.to_excel(xw, sheet_name="perspective_strict", index=False)
        perspective_conf.to_excel(xw, sheet_name="perspective_conf", index=False)
        by_marker_all.to_excel(xw, sheet_name="marker_all", index=False)
        by_marker_strict.to_excel(xw, sheet_name="marker_strict", index=False)
        by_marker_wide.to_excel(xw, sheet_name="marker_wide", index=False)
        by_marker_linked.to_excel(xw, sheet_name="marker_config_linked", index=False)
        marker_core.to_excel(xw, sheet_name="marker_core", index=False)
        marker_extended.to_excel(xw, sheet_name="marker_extended", index=False)
        marker_experimental.to_excel(xw, sheet_name="marker_experimental", index=False)
        by_precision.to_excel(xw, sheet_name="precision_layers", index=False)
        by_microtheme_all.to_excel(xw, sheet_name="microtheme_all", index=False)
        by_microtheme_linked.to_excel(xw, sheet_name="microtheme_linked", index=False)
        by_others_speechact.to_excel(xw, sheet_name="others_speechact", index=False)
        book_normalized.to_excel(xw, sheet_name="book_normalized", index=False)
        part_normalized.to_excel(xw, sheet_name="part_normalized", index=False)
        by_book_top.to_excel(xw, sheet_name="book_top_markers", index=False)
        by_part_top.to_excel(xw, sheet_name="part_top_markers", index=False)
        overlap_report.to_excel(xw, sheet_name="word_phrase_overlaps", index=False)
        phrase_family_coverage.to_excel(xw, sheet_name="phrase_family_coverage", index=False)
        nested_phrase_candidates.to_excel(xw, sheet_name="nested_phrase_report", index=False)
        perspective_microtheme_matrix.to_excel(xw, sheet_name="perspective_microtheme", index=False)
        part_microtheme_matrix.to_excel(xw, sheet_name="part_microtheme", index=False)
        core_extended_by_part.to_excel(xw, sheet_name="core_extended_by_part", index=False)
        inner_speech_by_part.to_excel(xw, sheet_name="inner_speech_by_part", index=False)
        external_pressure_by_part.to_excel(xw, sheet_name="external_pressure_by_part", index=False)
        hero_vs_corpus_microthemes.to_excel(xw, sheet_name="hero_vs_corpus_micro", index=False)
        melekhov_cognitive_profile.to_excel(xw, sheet_name="melekhov_profile", index=False)
        analytical_indices.to_excel(xw, sheet_name="analytical_indices", index=False)
        part_normalized.to_excel(xw, sheet_name="density_by_part", index=False)
        noise_report.to_excel(xw, sheet_name="noise_report", index=False)
        manual_check_sample.to_excel(xw, sheet_name="manual_check", index=False)

    if save_phrase_quality_report:
        build_phrase_quality_report(out_dir, hits_raw, markers)

    build_quality_dashboard(
        out_dir,
        hits_raw=hits_raw,
        counted=counted,
        markers=markers,
        noise_report=noise_report,
        nested_phrase_candidates=nested_phrase_candidates,
    )
    build_manual_verification(out_dir, hits_raw=hits_raw, counted=counted)

    return out_xlsx
