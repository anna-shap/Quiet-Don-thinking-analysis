from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from tdmm.utils.storage_utils import load_df


@dataclass
class ShowcaseConfig:
    top_n: int = 25
    examples_per_marker: int = 4


def _count_table(df: pd.DataFrame, group_cols: list[str], total_name: str = "count") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + [total_name, "share_pct"])
    out = df.groupby(group_cols, dropna=False).size().reset_index(name=total_name).sort_values(total_name, ascending=False)
    total = int(out[total_name].sum())
    out["share_pct"] = (out[total_name] / total * 100).round(2) if total else 0.0
    return out


def _merge_marker_meta(hits: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level", "marker_source", "source_type", "risk_comment"] if c in markers.columns]
    hits = hits.copy()
    markers = markers.copy()
    hits["marker_id"] = hits["marker_id"].astype(str)
    markers["marker_id"] = markers["marker_id"].astype(str)
    drop_cols = [c for c in cols if c != "marker_id" and c in hits.columns]
    if drop_cols:
        hits = hits.drop(columns=drop_cols)
    return hits.merge(markers[cols], on="marker_id", how="left")


def build_showcase(
    out_dir: str | Path,
    parts_csv: Optional[str | Path] = None,
    books_csv: Optional[str | Path] = None,
    cfg: ShowcaseConfig | None = None,
) -> Path:
    """Create an Excel pack for presentation, thesis tables, and manual review."""

    out_dir = Path(out_dir)
    cfg = cfg or ShowcaseConfig()

    markers = load_df(out_dir / "MARKERS_MASTER").copy()
    markers["marker_id"] = markers["marker_id"].astype(str)

    hits = load_df(out_dir / "TD_HITS_TAGGED").copy()
    hits = _merge_marker_meta(hits, markers)

    hits_clean = hits[hits.get("is_noise", False) != True].copy() if "is_noise" in hits.columns else hits.copy()  # noqa: E712
    if "count_in_main_stats" in hits_clean.columns:
        hits_counted = hits_clean[hits_clean["count_in_main_stats"] != False].copy()  # noqa: E712
    else:
        hits_counted = hits_clean.copy()

    strict = hits_counted[hits_counted.get("MelekhovLink", "").astype(str) == "YES"].copy()
    wide = hits_counted[hits_counted.get("MelekhovLink", "").astype(str).isin(["YES", "PROBABLE"])].copy()

    top_all = _count_table(hits_counted, ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level"]).head(cfg.top_n)
    top_strict = _count_table(strict, ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level"]).head(cfg.top_n)
    top_wide = _count_table(wide, ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level"]).head(cfg.top_n)
    micro_strict = _count_table(strict, ["microtheme"]).head(cfg.top_n)
    perspective_strict = _count_table(strict, ["Perspective"])
    perspective_wide = _count_table(wide, ["Perspective"])
    layers = _count_table(hits_counted, ["precision_level"])
    family_wide = _count_table(wide[wide.get("unit_type", "").astype(str).str.upper() == "PHRASE"], ["phrase_family", "microtheme", "precision_level"]).head(cfg.top_n) if "phrase_family" in wide.columns else pd.DataFrame()

    example_cols = [
        "hit_id", "marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "precision_level",
        "Book", "Part", "MelekhovLink", "MelekhovEvidence", "Perspective",
        "PerspectiveConfidence", "PerspectiveEvidence", "SpeakerDetected", "AddresseeDetected",
        "OthersSpeechAct", "target_surface", "sent_text", "context_window", "risk_comment",
    ]

    def _examples(source: pd.DataFrame, title_filter: str = "") -> pd.DataFrame:
        rows = []
        if source.empty:
            return pd.DataFrame(columns=example_cols + ["selection"])
        top_ids = _count_table(source, ["marker_id"]).head(cfg.top_n)["marker_id"].astype(str).tolist()
        for mid in top_ids:
            subset = source[source["marker_id"].astype(str) == mid].sort_values(["sent_id", "token_idx"]).head(cfg.examples_per_marker)
            for _, r in subset.iterrows():
                row = {c: r.get(c, "") for c in example_cols}
                row["selection"] = title_filter
                rows.append(row)
        return pd.DataFrame(rows)

    def _best_thesis_examples() -> pd.DataFrame:
        """Compact, high-value examples for direct use in the thesis text."""
        buckets = [
            ("повествователь о герое", wide[wide.get("Perspective", "") == "NARRATOR_ABOUT_HERO"].copy()),
            ("внутренняя речь героя", wide[wide.get("Perspective", "") == "HERO_INNER_SPEECH"].copy()),
            ("прямая речь героя", wide[wide.get("Perspective", "") == "HERO_DIRECT_SPEECH"].copy()),
            ("другие персонажи к герою", wide[wide.get("Perspective", "") == "OTHERS_TO_HERO"].copy()),
            ("память / воспоминание", wide[wide.get("microtheme", "").astype(str).str.contains("память|воспомин", case=False, regex=True)].copy()),
            ("понимание / ясность", wide[wide.get("microtheme", "").astype(str).str.contains("пониман|ясность|догад|соображ", case=False, regex=True)].copy()),
            ("решение / намерение", wide[wide.get("microtheme", "").astype(str).str.contains("решение|намерение", case=False, regex=True)].copy()),
            ("сомнение / колебание", wide[wide.get("microtheme", "").astype(str).str.contains("сомнен|колеб", case=False, regex=True)].copy()),
        ]
        rows = []
        for title, frame in buckets:
            if frame.empty:
                continue
            # Prefer high-confidence, counted, compact examples.
            ff = frame.copy()
            if "PerspectiveConfidence" in ff.columns:
                order_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                ff["_conf_order"] = ff["PerspectiveConfidence"].astype(str).str.upper().map(order_map).fillna(3)
            else:
                ff["_conf_order"] = 3
            ff = ff.sort_values(["_conf_order", "sent_id", "token_idx"]).head(2)
            for _, r in ff.iterrows():
                row = {c: r.get(c, "") for c in example_cols}
                row["thesis_bucket"] = title
                row["why_selected"] = "автоматически выбран как компактный пример для качественного разбора"
                rows.append(row)
        return pd.DataFrame(rows)

    best_thesis_examples = _best_thesis_examples()

    examples_strict = _examples(strict, "strict hero contexts")
    examples_core = _examples(wide[wide.get("precision_level", "").astype(str).str.upper() == "CORE"].copy(), "CORE")
    examples_extended = _examples(wide[wide.get("precision_level", "").astype(str).str.upper() == "EXTENDED"].copy(), "EXTENDED")
    examples_experimental = _examples(hits_clean[hits_clean.get("precision_level", "").astype(str).str.upper() == "EXPERIMENTAL"].copy(), "EXPERIMENTAL")
    examples_inner = _examples(wide[wide.get("Perspective", "") == "HERO_INNER_SPEECH"].copy(), "inner speech")
    examples_memory = _examples(wide[wide.get("microtheme", "").astype(str).str.contains("память|воспомин", case=False, regex=True)].copy(), "memory")
    examples_understanding = _examples(wide[wide.get("microtheme", "").astype(str).str.contains("пониман|ясность|догад|соображ", case=False, regex=True)].copy(), "understanding")
    examples_decision = _examples(wide[wide.get("microtheme", "").astype(str).str.contains("решение|намерение", case=False, regex=True)].copy(), "decision")
    examples_doubt = _examples(wide[wide.get("microtheme", "").astype(str).str.contains("сомнен|колеб", case=False, regex=True)].copy(), "doubt")
    examples_others_to = _examples(wide[wide.get("Perspective", "") == "OTHERS_TO_HERO"].copy(), "others to hero")
    examples_narr = _examples(strict[strict.get("Perspective", "") == "NARRATOR_ABOUT_HERO"].copy(), "narrator about hero")
    examples_others = _examples(wide[wide.get("Perspective", "").isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"])].copy(), "other characters")

    noise_examples = pd.DataFrame()
    if "is_noise" in hits.columns:
        noise_examples = hits[hits["is_noise"] == True].head(150).copy()  # noqa: E712

    dyn_parts = pd.DataFrame()
    if "Part" in strict.columns and not top_strict.empty:
        ids = top_strict["marker_id"].astype(str).tolist()
        sub = strict[strict["marker_id"].isin(ids)].copy()
        if len(sub):
            dyn_parts = sub.groupby(["Part", "marker_label"], dropna=False).size().reset_index(name="count")

    dyn_books = pd.DataFrame()
    if "Book" in strict.columns and not top_strict.empty:
        ids = top_strict["marker_id"].astype(str).tolist()
        sub = strict[strict["marker_id"].isin(ids)].copy()
        if len(sub):
            dyn_books = sub.groupby(["Book", "marker_label"], dropna=False).size().reset_index(name="count")

    out_xlsx = out_dir / "SHOWCASE.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame({
            "field": ["назначение", "строгая выборка", "широкая выборка", "слои маркеров", "где полный массив", "что проверять вручную"],
            "value": [
                "Витрина результатов: топы, микротемы, перспектива, динамика и контексты для ВКР/доклада.",
                "MelekhovLink = YES: имя героя найдено в том же предложении.",
                "MelekhovLink = YES + PROBABLE: имя героя найдено в том же или соседнем предложении.",
                "CORE — основной слой; EXTENDED — расширенный слой; EXPERIMENTAL — спорные кандидаты для ручной проверки.",
                "Полные таблицы: TD_HITS_TAGGED.parquet/.csv, STATS.xlsx, PHRASE_QUALITY_REPORT.xlsx.",
                "PROBABLE, UNKNOWN, фразовые маркеры, EXTENDED/EXPERIMENTAL и шумовые исключения.",
            ],
        }).to_excel(w, sheet_name="README", index=False)
        top_all.to_excel(w, sheet_name="TOP_ALL", index=False)
        top_strict.to_excel(w, sheet_name="TOP_MELEKHOV_YES", index=False)
        top_wide.to_excel(w, sheet_name="TOP_MELEKHOV_WIDE", index=False)
        layers.to_excel(w, sheet_name="PRECISION_LAYERS", index=False)
        family_wide.to_excel(w, sheet_name="PHRASE_FAMILIES", index=False)
        micro_strict.to_excel(w, sheet_name="MICROTHEMES_YES", index=False)
        perspective_strict.to_excel(w, sheet_name="PERSPECTIVE_YES", index=False)
        perspective_wide.to_excel(w, sheet_name="PERSPECTIVE_WIDE", index=False)
        dyn_parts.to_excel(w, sheet_name="DYNAMICS_PARTS", index=False)
        dyn_books.to_excel(w, sheet_name="DYNAMICS_BOOKS", index=False)
        best_thesis_examples.to_excel(w, sheet_name="BEST_THESIS_EXAMPLES", index=False)
        examples_strict.to_excel(w, sheet_name="EXAMPLES", index=False)
        examples_core.to_excel(w, sheet_name="CORE_EXAMPLES", index=False)
        examples_extended.to_excel(w, sheet_name="EXTENDED_EXAMPLES", index=False)
        examples_experimental.to_excel(w, sheet_name="EXPERIMENTAL", index=False)
        examples_inner.to_excel(w, sheet_name="INNER_SPEECH", index=False)
        examples_memory.to_excel(w, sheet_name="MEMORY", index=False)
        examples_understanding.to_excel(w, sheet_name="UNDERSTANDING", index=False)
        examples_decision.to_excel(w, sheet_name="DECISION", index=False)
        examples_doubt.to_excel(w, sheet_name="DOUBT", index=False)
        examples_others_to.to_excel(w, sheet_name="OTHERS_TO_HERO", index=False)
        examples_narr.to_excel(w, sheet_name="NARRATOR", index=False)
        examples_others.to_excel(w, sheet_name="OTHERS", index=False)
        noise_examples.to_excel(w, sheet_name="NOISE_EXAMPLES", index=False)

    return out_xlsx


def make_showcase(out_dir: str | Path) -> Path:
    return build_showcase(out_dir)
