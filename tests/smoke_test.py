from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run_pipeline(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "out_demo_sample"
    cmd = [
        sys.executable,
        "-m",
        "tdmm.run_pipeline",
        "--config",
        str(root / "config.json"),
        "--corpus",
        str(root / "data" / "corpus" / "quiet_don_SAMPLE.txt"),
        "--out",
        str(out),
        "--make-plots",
        "--max-sents",
        "50",
        "--clean-out",
    ]
    subprocess.run(cmd, check=True, cwd=root)
    return out


def test_smoke_pipeline_outputs_and_quality_flags(tmp_path: Path) -> None:
    out = _run_pipeline(tmp_path)

    required_files = [
        "MARKERS_MASTER.csv",
        "SENTS.csv",
        "HEADINGS.csv",
        "TD_HITS_TAGGED.csv",
        "STATS.xlsx",
        "SHOWCASE.xlsx",
        "PHRASE_QUALITY_REPORT.xlsx",
        "QUALITY_DASHBOARD.xlsx",
        "MANUAL_VERIFICATION.xlsx",
        "RUN_METADATA.json",
        "plot_part_microtheme_heatmap.png",
        "plot_perspective_microtheme_heatmap.png",
        "plot_top_phrase_families.png",
        "plot_melekhov_cognitive_profile.png",
    ]
    for name in required_files:
        assert (out / name).exists(), name

    markers = pd.read_csv(out / "MARKERS_MASTER.csv", encoding="utf-8-sig")
    hits = pd.read_csv(out / "TD_HITS_TAGGED.csv", encoding="utf-8-sig")
    sents = pd.read_csv(out / "SENTS.csv", encoding="utf-8-sig")
    headings = pd.read_csv(out / "HEADINGS.csv", encoding="utf-8-sig")
    meta = json.loads((out / "RUN_METADATA.json").read_text(encoding="utf-8"))
    overview = pd.read_excel(out / "STATS.xlsx", sheet_name="overview")
    overview_map = dict(zip(overview["metric"], overview["value"]))
    stats_sheets = pd.ExcelFile(out / "STATS.xlsx").sheet_names
    showcase_sheets = pd.ExcelFile(out / "SHOWCASE.xlsx").sheet_names
    manual_sheets = pd.ExcelFile(out / "MANUAL_VERIFICATION.xlsx").sheet_names

    assert len(markers[markers["unit_type"].astype(str).str.upper() == "PHRASE"]) >= 600
    assert {"CORE", "EXTENDED"}.issubset(set(markers["precision_level"].astype(str)))
    assert "phrase_family" in markers.columns
    assert "память / воспоминание" in set(markers["microtheme"].astype(str))
    assert meta["count_overlapping_word_hits"] is False
    assert meta["phrase_overlap_mode"] == "prefer_longest"
    assert int(meta["hits_counted_main"]) == int(overview_map["hits_total_counted"])
    assert "melekhov_profile" in stats_sheets
    assert "analytical_indices" in stats_sheets
    assert "BEST_THESIS_EXAMPLES" in showcase_sheets
    assert {"hit_check", "phrase_check", "perspective_check", "probable_link_check", "noise_check"}.issubset(set(manual_sheets))
    assert not str(meta["config_path"]).startswith("/")

    # Headings must be detected and kept out of analytical sentences.
    assert not headings.empty
    assert not sents["sent_text"].astype(str).str.contains("Книга первая", regex=False).any()

    # WORD inside PHRASE remains auditable, but is removed from main accounting.
    overlap_words = hits[(hits["hit_type"].astype(str).str.upper() == "WORD") & (hits["overlaps_phrase"] == True)]  # noqa: E712
    assert not overlap_words.empty
    assert (overlap_words["count_in_main_stats"] == False).any()  # noqa: E712

    assert {"PerspectiveEvidence", "PerspectiveConfidence", "SpeakerDetected", "AddresseeDetected"}.issubset(hits.columns)
    assert "HERO_INNER_SPEECH" in set(hits["Perspective"].astype(str))
    pod = hits[hits["target_surface"].astype(str).str.lower() == "подумаешь"]
    assert not pod.empty
    assert (pod["is_noise"] == True).all()  # noqa: E712


def test_validate_project_runs_without_temp_leftover(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "tdmm.validate_project",
        "--config",
        str(root / "config.json"),
        "--corpus",
        str(root / "data" / "corpus" / "quiet_don_SAMPLE.txt"),
    ]
    subprocess.run(cmd, check=True, cwd=root)
    assert not (root / "out_validate_tmp").exists()
