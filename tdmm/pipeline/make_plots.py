from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from tdmm.utils.storage_utils import load_df


PERSPECTIVE_LABELS_RU: Dict[str, str] = {
    "HERO_INNER_SPEECH": "герой: внутренняя речь",
    "HERO_DIRECT_SPEECH": "герой: прямая речь",
    "NARRATOR_ABOUT_HERO": "повествователь о герое",
    "OTHERS_TO_HERO": "другие обращаются к герою",
    "OTHERS_ABOUT_HERO": "другие говорят о герое",
    "OTHERS_NEAR_HERO": "другие рядом с героем",
    "UNKNOWN": "неопределено",
}
PERSPECTIVE_ORDER: Sequence[str] = ("HERO_INNER_SPEECH", "HERO_DIRECT_SPEECH", "NARRATOR_ABOUT_HERO", "OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO", "UNKNOWN")


def _safe_savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    try:
        plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] SVG plot skipped for {path.name}: {exc}")
    plt.close()


def _linked_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    mode = (mode or "WIDE").upper()
    if mode == "STRICT":
        return df["MelekhovLink"].astype(str) == "YES"
    return df["MelekhovLink"].astype(str).isin(["YES", "PROBABLE"])


def _counts_for(df_hits: pd.DataFrame, id2label: Dict[str, str]) -> pd.DataFrame:
    if df_hits.empty:
        return pd.DataFrame(columns=["marker_id", "count", "marker_label"])
    c = df_hits.groupby("marker_id").size().reset_index(name="count")
    c["marker_id"] = c["marker_id"].astype(str)
    c["marker_label"] = c["marker_id"].map(id2label).fillna(c["marker_id"])
    return c.sort_values("count", ascending=False)


def _barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, filename: Path, xlabel: str = "количество", top_n: int = 15) -> None:
    if df.empty:
        return
    d = df.head(top_n).copy()
    plt.figure(figsize=(12, 6.2))
    plt.barh(d[label_col].astype(str), d[value_col])
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)
    _safe_savefig(filename)




def _matrix_heatmap(df: pd.DataFrame, index_col: str, col_col: str, title: str, filename: Path) -> None:
    if df.empty or index_col not in df.columns or col_col not in df.columns:
        return
    mat = df.groupby([index_col, col_col], dropna=False).size().reset_index(name="count")
    pivot = mat.pivot(index=index_col, columns=col_col, values="count").fillna(0)
    if pivot.empty:
        return
    # Keep the figure readable on full corpus: show the most active columns first.
    col_order = pivot.sum(axis=0).sort_values(ascending=False).head(14).index.tolist()
    pivot = pivot[col_order]
    fig_w = max(10, min(18, 1.1 * len(pivot.columns) + 4))
    fig_h = max(5.5, min(12, 0.55 * len(pivot.index) + 3))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.colorbar(label="количество")
    plt.title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            if val > 0:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=8)
    _safe_savefig(filename)


def _grouped_lines(df: pd.DataFrame, index_col: str, col_col: str, title: str, filename: Path) -> None:
    if df.empty or index_col not in df.columns or col_col not in df.columns:
        return
    pivot = df.groupby([index_col, col_col], dropna=False).size().reset_index(name="count").pivot(index=index_col, columns=col_col, values="count").fillna(0)
    if pivot.empty:
        return
    plt.figure(figsize=(11.5, 6))
    for col in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[col].values, marker="o", label=str(col))
    plt.title(title)
    plt.xlabel("часть")
    plt.ylabel("количество")
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="upper left", fontsize=9)
    _safe_savefig(filename)

def _make_autopct(values: Sequence[float]):
    total = float(sum(values)) if values else 0.0

    def inner(pct: float) -> str:
        if total <= 0 or pct < 3:
            return ""
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({absolute})"

    return inner


def _pie(labels: Sequence[str], values: Sequence[float], title: str, filename: Path) -> None:
    if not values or sum(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    wedges, _texts, _autotexts = ax.pie(values, labels=None, autopct=_make_autopct(values), startangle=90, pctdistance=0.75)
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title(title)
    ax.axis("equal")
    ax.legend(wedges, labels, title="категории", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10)
    _safe_savefig(filename)


def plot_top_markers_pie(counts_df: pd.DataFrame, out_dir: Path, *, title: str, filename: str, top_n: int = 5) -> None:
    df = counts_df.sort_values("count", ascending=False).copy()
    if df.empty:
        return
    top = df.head(top_n).copy()
    rest = int(df.iloc[top_n:]["count"].sum())
    labels = top["marker_label"].astype(str).tolist()
    values = top["count"].astype(float).tolist()
    if rest > 0:
        labels.append("прочие")
        values.append(float(rest))
    _pie(labels, values, title, out_dir / filename)


def plot_perspective(hits_linked: pd.DataFrame, out_dir: Path) -> None:
    if hits_linked.empty:
        return
    df = hits_linked.copy()
    df["Perspective"] = df["Perspective"].fillna("UNKNOWN").astype(str)
    counts = df.groupby("Perspective").size().reindex(list(PERSPECTIVE_ORDER), fill_value=0).reset_index(name="count")
    counts["label_ru"] = counts["Perspective"].map(PERSPECTIVE_LABELS_RU).fillna(counts["Perspective"])
    plt.figure(figsize=(11.5, 5.4))
    bars = plt.bar(counts["label_ru"], counts["count"])
    plt.ylabel("количество контекстов")
    plt.xlabel("речевая перспектива")
    plt.title("Речевая перспектива в контекстах Григория Мелехова")
    plt.xticks(rotation=12, ha="right")
    for b in bars:
        h = b.get_height()
        if h > 0:
            plt.annotate(f"{int(h)}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    _safe_savefig(out_dir / "plot_perspective.png")


def plot_perspective_pie(hits_linked: pd.DataFrame, out_dir: Path) -> None:
    if hits_linked.empty:
        return
    counts = hits_linked.groupby("Perspective").size().reindex(list(PERSPECTIVE_ORDER), fill_value=0).reset_index(name="count")
    counts = counts[counts["count"] > 0]
    labels = counts["Perspective"].map(PERSPECTIVE_LABELS_RU).fillna(counts["Perspective"]).astype(str).tolist()
    values = counts["count"].astype(float).tolist()
    _pie(labels, values, "Доли речевых перспектив", out_dir / "plot_perspective_pie.png")


def plot_dynamics_top_markers(hits_linked: pd.DataFrame, id2label: Dict[str, str], seg_csv: str | Path, out_dir: Path, *, top_marker_ids: Sequence[str], seg_col: str, seg_name_col: str, filename: str, title: str) -> None:
    if hits_linked.empty or not top_marker_ids:
        return
    seg = pd.read_csv(seg_csv)
    order = seg.sort_values(seg_col)[seg_name_col].astype(str).tolist() if seg_col in seg.columns and seg_name_col in seg.columns else []
    df = hits_linked[hits_linked["marker_id"].astype(str).isin([str(x) for x in top_marker_ids])].copy()
    if df.empty or seg_name_col.replace("_name", "").capitalize() not in df.columns:
        pass

    plot_col = "Book" if "book" in seg_name_col else "Part"
    if plot_col not in df.columns:
        return
    pivot = df.groupby([plot_col, "marker_id"]).size().reset_index(name="count").pivot(index=plot_col, columns="marker_id", values="count").fillna(0)
    if order:
        pivot = pivot.reindex(order).fillna(0)
    plt.figure(figsize=(12, 6.2))
    any_series = False
    for mid in [str(x) for x in top_marker_ids]:
        if mid not in pivot.columns:
            continue
        any_series = True
        plt.plot(pivot.index.astype(str), pivot[mid].values, marker="o", label=id2label.get(mid, mid))
    plt.title(title)
    plt.xlabel("сегмент текста")
    plt.ylabel("количество")
    plt.xticks(rotation=15, ha="right")
    if any_series:
        plt.legend(loc="upper left", fontsize=9)
    _safe_savefig(out_dir / filename)


def make_plots(
    out_dir: str | Path,
    parts_csv: Optional[str | Path] = None,
    books_csv: Optional[str | Path] = None,
    *,
    include_noise_in_counts: bool = False,
    melekhov_link_mode: str = "WIDE",
) -> None:
    out = Path(out_dir)
    hits_raw = load_df(out / "TD_HITS_TAGGED").copy()
    markers = load_df(out / "MARKERS_MASTER").copy()
    hits_raw["marker_id"] = hits_raw["marker_id"].astype(str)
    markers["marker_id"] = markers["marker_id"].astype(str)

    hits = hits_raw.copy()
    if ("is_noise" in hits.columns) and (not include_noise_in_counts):
        hits = hits[hits["is_noise"] != True].copy()  # noqa: E712
    if "count_in_main_stats" in hits.columns:
        hits = hits[hits["count_in_main_stats"] != False].copy()  # noqa: E712

    id2label = dict(zip(markers["marker_id"], markers.get("marker_label", markers.get("unit_norm", markers["marker_id"]))))
    id2micro = dict(zip(markers["marker_id"], markers.get("microtheme", "")))
    id2family = dict(zip(markers["marker_id"], markers.get("phrase_family", "")))

    linked = hits[_linked_mask(hits, melekhov_link_mode)].copy()
    linked_counts = _counts_for(linked, id2label)

    _barh(_counts_for(hits, id2label), "marker_label", "count", "Топ маркеров мышления: весь корпус", out / "plot_top_markers_all.png")
    _barh(linked_counts, "marker_label", "count", "Топ маркеров мышления в контекстах Григория Мелехова", out / "plot_top_markers_melekhov.png")
    plot_top_markers_pie(linked_counts, out, title="Доли топ-маркеров в контекстах Григория Мелехова", filename="plot_top_markers_melekhov_pie.png")

    if not linked.empty:
        tmp = linked.copy()
        tmp["microtheme"] = tmp["marker_id"].map(id2micro).fillna("")
        micro = tmp.groupby("microtheme").size().reset_index(name="count").sort_values("count", ascending=False)
        _barh(micro, "microtheme", "count", "Микротемы маркеров в контекстах Григория Мелехова", out / "plot_microthemes_melekhov.png")
        _barh(micro, "microtheme", "count", "Когнитивный профиль Григория Мелехова", out / "plot_melekhov_cognitive_profile.png")
        if "precision_level" in tmp.columns:
            layers = tmp.groupby("precision_level").size().reset_index(name="count").sort_values("count", ascending=False)
            _barh(layers, "precision_level", "count", "Слои маркеров CORE / EXTENDED / EXPERIMENTAL", out / "plot_phrase_layers.png", top_n=5)
        if "phrase_family" not in tmp.columns:
            tmp["phrase_family"] = tmp["marker_id"].map(id2family).fillna("")
        fam = tmp[tmp["hit_type"].astype(str).str.upper() == "PHRASE"].groupby("phrase_family").size().reset_index(name="count").sort_values("count", ascending=False) if "hit_type" in tmp.columns else pd.DataFrame()
        _barh(fam, "phrase_family", "count", "Топ семейств фразовых маркеров", out / "plot_top_phrase_families.png")

        _matrix_heatmap(tmp, "Perspective", "microtheme", "Матрица: речевая перспектива × микротема", out / "plot_perspective_microtheme_heatmap.png")
        _matrix_heatmap(tmp, "Part", "microtheme", "Матрица: часть текста × микротема", out / "plot_part_microtheme_heatmap.png")
        _grouped_lines(tmp, "Part", "precision_level", "Динамика слоёв CORE / EXTENDED по частям", out / "plot_core_extended_by_part.png")

        if "Part" in tmp.columns and "Perspective" in tmp.columns:
            part_total = tmp.groupby("Part").size().reset_index(name="all_hits")
            inner = tmp[tmp["Perspective"].astype(str) == "HERO_INNER_SPEECH"].groupby("Part").size().reset_index(name="inner_hits")
            inner = part_total.merge(inner, on="Part", how="left").fillna(0)
            inner["share_pct"] = (inner["inner_hits"] / inner["all_hits"] * 100).round(2)
            _barh(inner.sort_values("share_pct", ascending=False), "Part", "share_pct", "Доля внутренней речи героя по частям", out / "plot_inner_speech_share_by_part.png", xlabel="доля, %", top_n=20)
            ext = tmp[tmp["Perspective"].astype(str).isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"])].groupby("Part").size().reset_index(name="external_hits")
            ext = part_total.merge(ext, on="Part", how="left").fillna(0)
            ext["share_pct"] = (ext["external_hits"] / ext["all_hits"] * 100).round(2)
            _barh(ext.sort_values("share_pct", ascending=False), "Part", "share_pct", "Доля чужой речи о/к герою по частям", out / "plot_external_pressure_by_part.png", xlabel="доля, %", top_n=20)

    plot_perspective(linked, out)
    plot_perspective_pie(linked, out)

    strict = hits[hits["MelekhovLink"].astype(str) == "YES"].copy()
    wide = hits[hits["MelekhovLink"].astype(str).isin(["YES", "PROBABLE"])].copy()
    sw = pd.DataFrame({"mode": ["STRICT: YES", "WIDE: YES+PROBABLE"], "count": [len(strict), len(wide)]})
    _barh(sw, "mode", "count", "Строгая и широкая привязка к герою", out / "plot_strict_vs_wide.png", top_n=2)

    if "is_noise" in hits_raw.columns:
        noise = hits_raw[hits_raw["is_noise"] == True].copy()  # noqa: E712
        if not noise.empty:
            nr = noise.groupby("noise_reason").size().reset_index(name="count").sort_values("count", ascending=False)
            _barh(nr, "noise_reason", "count", "Исключённые шумовые случаи", out / "plot_noise_report.png")

    if "microtheme" not in hits.columns:
        hits["microtheme"] = hits["marker_id"].map(id2micro).fillna("")
    if not hits.empty and not linked.empty:
        corp = hits.groupby("microtheme").size().reset_index(name="corpus_count")
        hero = linked.groupby("microtheme").size().reset_index(name="melekhov_count")
        cmp = corp.merge(hero, on="microtheme", how="outer").fillna(0)
        cmp["difference_abs"] = cmp["melekhov_count"] - cmp["corpus_count"]
        # show Melekhov distribution as the directly interpretable chart
        _barh(cmp.sort_values("melekhov_count", ascending=False), "microtheme", "melekhov_count", "Микротемы: контексты Мелехова на фоне корпуса", out / "plot_corpus_vs_melekhov_microthemes.png")

    if parts_csv is not None and Path(parts_csv).exists():
        try:
            seg = pd.read_csv(parts_csv)
            rows = []
            for _, r in seg.iterrows():
                start, end = int(r["start_sent"]), int(r["end_sent"])
                part = str(r.get("part_name", r.get("part_id", "UNKNOWN")))
                word_n = 0
                if "word_count" in load_df(out / "SENTS").columns:
                    ss = load_df(out / "SENTS")
                    word_n = int(ss[(ss["sent_id"].astype(int) >= start) & (ss["sent_id"].astype(int) <= end)]["word_count"].sum())
                hits_n = int(((linked["sent_id"].astype(int) >= start) & (linked["sent_id"].astype(int) <= end)).sum()) if not linked.empty else 0
                rows.append({"Part": part, "hits_per_10000_words": round(hits_n / word_n * 10000, 3) if word_n else 0.0})
            dens = pd.DataFrame(rows)
            _barh(dens.sort_values("hits_per_10000_words", ascending=False), "Part", "hits_per_10000_words", "Когнитивная плотность по частям", out / "plot_cognitive_density_by_part.png", xlabel="маркеров на 10000 слов", top_n=20)
        except Exception:
            pass

    top_ids = linked_counts.head(6)["marker_id"].astype(str).tolist()
    if parts_csv is not None and Path(parts_csv).exists() and top_ids:
        plot_dynamics_top_markers(linked, id2label, parts_csv, out, top_marker_ids=top_ids, seg_col="part_id", seg_name_col="part_name", filename="plot_dynamics_top_markers.png", title="Динамика топ-маркеров по частям текста")
    if books_csv is not None and Path(books_csv).exists() and top_ids:
        plot_dynamics_top_markers(linked, id2label, books_csv, out, top_marker_ids=top_ids, seg_col="book_id", seg_name_col="book_name", filename="plot_dynamics_top_markers_books.png", title="Динамика топ-маркеров по книгам романа")
