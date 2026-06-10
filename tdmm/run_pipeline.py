from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from tdmm.build_parts import build_books_and_parts
from tdmm.pipeline.build_markers import build_markers_master
from tdmm.pipeline.extract_hits import extract_corpus_hits
from tdmm.pipeline.make_plots import make_plots
from tdmm.pipeline.make_showcase import build_showcase
from tdmm.pipeline.make_stats import make_stats
from tdmm.pipeline.tag_hits import tag_hits
from tdmm.utils.config_utils import Config, project_root_from_any
from tdmm.utils.storage_utils import load_df


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TD-MindMarkers pipeline")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--corpus", default=None, help="override corpus path")
    ap.add_argument("--out", default=None, help="override output directory")
    ap.add_argument("--make-plots", action="store_true", help="build plots")
    ap.add_argument("--max-sents", type=int, default=None, help="limit number of sentences (smoke runs)")
    ap.add_argument("--clean-out", action="store_true", help="remove output directory before running")
    ap.add_argument("--auto-build-segmentation", action="store_true", help="build missing books/parts CSV from explicit corpus headings")
    ap.add_argument("--make_plots", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max_sents", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _save_run_metadata(*, out_dir: Path, cfg: Config, config_path: Path, corpus: Path, parts_csv: Path | None, books_csv: Path | None) -> None:
    root = config_path.resolve().parent
    meta: Dict[str, Any] = {
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "project_name": cfg.project_name,
        "analysis_profile": cfg.analysis_profile,
        "corpus_path": _rel_path(corpus, root),
        "corpus_sha256": _sha256(corpus),
        "config_path": _rel_path(config_path, root),
        "config_sha256": _sha256(config_path),
        "parts_csv": _rel_path(parts_csv, root) if parts_csv else "",
        "books_csv": _rel_path(books_csv, root) if books_csv else "",
        "melekhov_link_mode": cfg.melekhov_link_mode,
        "include_noise_in_counts": cfg.include_noise_in_counts,
        "count_overlapping_word_hits": cfg.count_overlapping_word_hits,
        "enabled_precision_levels": cfg.enabled_precision_levels,
        "experimental_phrases_in_main_stats": cfg.experimental_phrases_in_main_stats,
        "min_perspective_confidence_main": cfg.min_perspective_confidence_main,
        "phrase_overlap_mode": cfg.phrase_overlap_mode,
        "remove_headings_from_analysis": cfg.remove_headings_from_analysis,
    }
    try:
        markers = load_df(out_dir / "MARKERS_MASTER")
        hits = load_df(out_dir / "TD_HITS_TAGGED")
        counted = hits.copy()
        if not cfg.include_noise_in_counts and "is_noise" in counted.columns:
            counted = counted[counted["is_noise"] != True].copy()  # noqa: E712
        if "count_in_main_stats" in counted.columns:
            counted = counted[counted["count_in_main_stats"] != False].copy()  # noqa: E712
        if cfg.enabled_precision_levels and "precision_level" in counted.columns:
            allowed = {str(x).upper() for x in cfg.enabled_precision_levels}
            counted = counted[counted["precision_level"].astype(str).str.upper().isin(allowed)].copy()
        if not cfg.experimental_phrases_in_main_stats and "precision_level" in counted.columns:
            counted = counted[counted["precision_level"].astype(str).str.upper() != "EXPERIMENTAL"].copy()
        conf_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        min_rank = conf_rank.get(str(cfg.min_perspective_confidence_main).upper(), 1)
        if "PerspectiveConfidence" in counted.columns:
            ranks = counted["PerspectiveConfidence"].astype(str).str.upper().map(conf_rank).fillna(1).astype(int)
            counted = counted[ranks >= min_rank].copy()
        meta.update({
            "markers_count": int(len(markers)),
            "phrase_markers_count": int((markers["unit_type"].astype(str).str.upper() == "PHRASE").sum()),
            "core_markers_count": int((markers.get("precision_level", "").astype(str).str.upper() == "CORE").sum()),
            "extended_markers_count": int((markers.get("precision_level", "").astype(str).str.upper() == "EXTENDED").sum()),
            "experimental_markers_count": int((markers.get("precision_level", "").astype(str).str.upper() == "EXPERIMENTAL").sum()),
            "hits_total_raw": int(len(hits)),
            "hits_counted_main": int(len(counted)),
        })
    except Exception as exc:  # noqa: BLE001
        meta["metadata_warning"] = str(exc)
    (out_dir / "RUN_METADATA.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    cfg = Config.load(args.config)
    config_path = Path(args.config).resolve()
    root = project_root_from_any(args.config)

    corpus = Path(args.corpus) if args.corpus else (root / cfg.corpus_path)
    out_dir = Path(args.out) if args.out else (root / cfg.output_dir)
    if args.clean_out and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plots_flag = bool(args.make_plots)
    max_sents = args.max_sents

    s1_csv = root / cfg.markers["s1_csv"]
    alex_txt = root / cfg.markers["alexandrova_txt"]
    phrases_csv = root / cfg.markers["phrases_csv"]
    exclusions_csv = root / cfg.markers.get("exclusions_csv", "") if cfg.markers.get("exclusions_csv") else None
    microthemes_csv = root / cfg.microthemes_csv if cfg.microthemes_csv else None

    parts_csv = root / cfg.segmentation.get("parts_csv", "") if cfg.segmentation else None
    books_csv = root / cfg.segmentation.get("books_csv", "") if cfg.segmentation else None

    if args.auto_build_segmentation or cfg.auto_build_segmentation:
        try:
            build_books_and_parts(
                str(corpus),
                out_books_csv=str(books_csv) if books_csv and str(books_csv) and not Path(books_csv).exists() else None,
                out_parts_csv=str(parts_csv) if parts_csv and str(parts_csv) and not Path(parts_csv).exists() else None,
                encoding_fallbacks=cfg.encoding_fallbacks,
                exclude_headers=True,
                exclude_epigraph=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] segmentation auto-build failed: {e}")

    if parts_csv and not Path(parts_csv).exists():
        parts_csv = None
    if books_csv and not Path(books_csv).exists():
        books_csv = None

    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}. Add full quiet_don.txt or pass --corpus data/corpus/quiet_don_SAMPLE.txt")

    print(f"Project: {cfg.project_name}")
    print(f"Profile: {cfg.analysis_profile}")
    print(f"Corpus : {corpus}")
    print(f"Out    : {out_dir}")
    print(f"Segm.  : books={books_csv if books_csv else 'none'}; parts={parts_csv if parts_csv else 'none'}")

    print("[1/6] Building MARKERS_MASTER")
    build_markers_master(
        s1_csv=s1_csv,
        alexandrova_txt=alex_txt,
        phrases_csv=phrases_csv,
        out_path=out_dir / "MARKERS_MASTER",
        export_csv=cfg.export_csv,
        csv_max_rows=cfg.csv_max_rows,
        enabled_precision_levels=cfg.enabled_precision_levels if not cfg.experimental_phrases_in_main_stats else None,
        microthemes_csv=microthemes_csv,
    )

    print("[2/6] Extracting hits from corpus")
    extract_corpus_hits(
        corpus_path=corpus,
        markers_master_path=out_dir / "MARKERS_MASTER",
        out_dir=out_dir,
        encoding_fallbacks=cfg.encoding_fallbacks,
        phrase_gap=cfg.phrase_gap,
        max_hits_sample=cfg.max_hits_sample,
        max_sents=max_sents,
        noise_filters=cfg.noise_filters,
        exclusions_csv=exclusions_csv,
        context_window_sentences=cfg.context_window_sentences,
        count_overlapping_word_hits=cfg.count_overlapping_word_hits,
        remove_headings_from_analysis=cfg.remove_headings_from_analysis,
        phrase_overlap_mode=cfg.phrase_overlap_mode,
        export_csv=cfg.export_csv,
        csv_max_rows=cfg.csv_max_rows,
    )

    print("[3/6] Tagging hits")
    tag_hits(
        out_dir=out_dir,
        name_variants=cfg.name_variants,
        window_sentences=cfg.window_sentences,
        parts_csv=parts_csv,
        books_csv=books_csv,
        max_hits_sample=cfg.max_hits_sample,
        export_csv=cfg.export_csv,
        csv_max_rows=cfg.csv_max_rows,
    )

    print("[4/6] Building STATS.xlsx and PHRASE_QUALITY_REPORT.xlsx")
    make_stats(
        out_dir=out_dir,
        segmentation_parts_csv=parts_csv,
        segmentation_books_csv=books_csv,
        include_noise_in_counts=cfg.include_noise_in_counts,
        melekhov_link_mode=cfg.melekhov_link_mode,
        analysis_profile=cfg.analysis_profile,
        count_overlapping_word_hits=cfg.count_overlapping_word_hits,
        enabled_precision_levels=cfg.enabled_precision_levels,
        experimental_phrases_in_main_stats=cfg.experimental_phrases_in_main_stats,
        save_phrase_quality_report=cfg.save_phrase_quality_report,
        min_perspective_confidence_main=cfg.min_perspective_confidence_main,
    )

    print("[5/6] Building SHOWCASE.xlsx")
    build_showcase(out_dir=out_dir, parts_csv=parts_csv, books_csv=books_csv)

    if make_plots_flag:
        print("[6/6] Making plots")
        make_plots(
            out_dir=out_dir,
            parts_csv=parts_csv,
            books_csv=books_csv,
            include_noise_in_counts=cfg.include_noise_in_counts,
            melekhov_link_mode=cfg.melekhov_link_mode,
        )
    else:
        print("[6/6] Plots skipped (use --make-plots)")

    if cfg.save_run_metadata:
        _save_run_metadata(out_dir=out_dir, cfg=cfg, config_path=config_path, corpus=corpus, parts_csv=parts_csv, books_csv=books_csv)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
