from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from tdmm.pipeline.build_markers import build_markers_master
from tdmm.utils.config_utils import Config, project_root_from_any
from tdmm.utils.io_utils import read_text_with_fallback
from tdmm.utils.text_utils import sentenize_text, split_headings_from_text
from tdmm.utils.storage_utils import load_df


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate TD-MindMarkers project inputs")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--keep-temp", action="store_true", help="keep out_validate_tmp after validation")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    root = project_root_from_any(args.config)
    corpus = Path(args.corpus) if args.corpus else root / cfg.corpus_path

    print(f"Project: {cfg.project_name}")
    print(f"Profile: {cfg.analysis_profile}")
    print(f"Corpus : {corpus}")
    if not corpus.exists():
        print("ERROR: corpus file is absent. Full run requires data/corpus/quiet_don.txt or explicit --corpus.")
        return 2

    text, enc = read_text_with_fallback(corpus, cfg.encoding_fallbacks)
    text_for_analysis, heading_rows = split_headings_from_text(text, remove_headings=cfg.remove_headings_from_analysis)
    sents = list(sentenize_text(text_for_analysis))
    print(f"Encoding used: {enc}")
    print(f"Sentences   : {len(sents)}")
    print(f"Headings    : {len(heading_rows)} detected; remove_headings_from_analysis={cfg.remove_headings_from_analysis}")
    if len(sents) < 10:
        _warn("very few sentences; probably sample or malformed corpus")

    for name, rel in cfg.markers.items():
        p = root / rel
        print(f"Marker file {name}: {'OK' if p.exists() else 'MISSING'} — {p}")

    for name, rel in cfg.segmentation.items():
        p = root / rel
        print(f"Segmentation {name}: {'OK' if p.exists() else 'MISSING'} — {p}")
        if not p.exists() and name == "books_csv":
            _warn("books.csv is absent; analysis by books will be skipped, analysis by parts remains valid")
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"  rows={len(df)}, cols={list(df.columns)}")
                if "start_sent" not in df.columns or "end_sent" not in df.columns:
                    _warn(f"{name} has no start_sent/end_sent columns")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR reading CSV: {exc}")

    tmp_dir = root / "out_validate_tmp"
    tmp = tmp_dir / "MARKERS_MASTER"
    try:
        build_markers_master(
            s1_csv=root / cfg.markers["s1_csv"],
            alexandrova_txt=root / cfg.markers["alexandrova_txt"],
            phrases_csv=root / cfg.markers["phrases_csv"],
            out_path=tmp,
            export_csv=True,
            enabled_precision_levels=None,
            microthemes_csv=(root / cfg.microthemes_csv) if cfg.microthemes_csv else None,
        )
        markers = load_df(tmp)
        print(f"Markers total: {len(markers)}")
        print(f"Phrase markers: {(markers['unit_type'].astype(str).str.upper() == 'PHRASE').sum()}")
        if "marker_id" in markers.columns and markers["marker_id"].duplicated().any():
            _warn("duplicated marker_id values found")
        if "unit_norm" in markers.columns and markers.duplicated(["unit_type", "unit_norm"]).any():
            _warn("duplicated unit_type+unit_norm values found after build")
        if "microtheme" in markers.columns and (markers["microtheme"].fillna("").astype(str).str.strip() == "").any():
            _warn("some markers have empty microtheme")
        if "precision_level" in markers.columns:
            print("Precision levels:")
            print(markers["precision_level"].value_counts(dropna=False).to_string())
            bad_levels = set(markers["precision_level"].astype(str).str.upper()) - {"CORE", "EXTENDED", "EXPERIMENTAL"}
            if bad_levels:
                _warn(f"unknown precision levels: {sorted(bad_levels)}")
        if "phrase_family" in markers.columns:
            empty_family = int((markers["phrase_family"].fillna("").astype(str).str.strip() == "").sum())
            if empty_family:
                _warn(f"empty phrase_family values: {empty_family}")
    finally:
        if tmp_dir.exists() and not args.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Validation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
