from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class Config:
    project_name: str
    corpus_path: str
    name_variants: List[str]
    window_sentences: int
    phrase_gap: int
    max_hits_sample: int
    encoding_fallbacks: List[str]
    csv_separators: List[str]
    markers: Dict[str, str]
    segmentation: Dict[str, str]
    output_dir: str

    export_csv: bool = True
    csv_max_rows: int = 200_000
    context_window_sentences: int = 1

    include_noise_in_counts: bool = False
    noise_filters: Dict[str, Any] = field(default_factory=lambda: {"podumayesh_particle": True})

    # STRICT -> only YES; WIDE -> YES + PROBABLE.
    melekhov_link_mode: str = "WIDE"

    # Diploma profile / quality-control switches.
    analysis_profile: str = "DIPLOMA_STRICT"
    count_overlapping_word_hits: bool = False
    enabled_precision_levels: List[str] = field(default_factory=lambda: ["CORE", "EXTENDED"])
    experimental_phrases_in_main_stats: bool = False
    save_phrase_quality_report: bool = True
    save_run_metadata: bool = True
    phrase_overlap_mode: str = "prefer_longest"
    microthemes_csv: str = ""
    remove_headings_from_analysis: bool = True
    use_existing_segmentation: bool = True
    auto_build_segmentation: bool = False
    min_perspective_confidence_main: str = "LOW"

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        return Config(
            project_name=str(raw["project_name"]),
            corpus_path=str(raw["corpus_path"]),
            name_variants=list(raw.get("name_variants", [])),
            window_sentences=int(raw.get("window_sentences", 1)),
            phrase_gap=int(raw.get("phrase_gap", 2)),
            max_hits_sample=int(raw.get("max_hits_sample", 500)),
            encoding_fallbacks=list(raw.get("encoding_fallbacks", ["utf-8", "utf-8-sig", "cp1251"])),
            csv_separators=list(raw.get("csv_separators", [",", ";", "\t"])),
            markers=dict(raw.get("markers", {})),
            segmentation=dict(raw.get("segmentation", {})),
            output_dir=str(raw.get("output_dir", "out_final")),
            export_csv=bool(raw.get("export_csv", True)),
            csv_max_rows=int(raw.get("csv_max_rows", 200_000)),
            context_window_sentences=int(raw.get("context_window_sentences", 1)),
            include_noise_in_counts=bool(raw.get("include_noise_in_counts", False)),
            noise_filters=dict(raw.get("noise_filters", {"podumayesh_particle": True})),
            melekhov_link_mode=str(raw.get("melekhov_link_mode", "WIDE")).upper(),
            analysis_profile=str(raw.get("analysis_profile", "DIPLOMA_STRICT")),
            count_overlapping_word_hits=bool(raw.get("count_overlapping_word_hits", False)),
            enabled_precision_levels=[str(x).upper() for x in raw.get("enabled_precision_levels", ["CORE", "EXTENDED"])],
            experimental_phrases_in_main_stats=bool(raw.get("experimental_phrases_in_main_stats", False)),
            save_phrase_quality_report=bool(raw.get("save_phrase_quality_report", True)),
            save_run_metadata=bool(raw.get("save_run_metadata", True)),
            phrase_overlap_mode=str(raw.get("phrase_overlap_mode", "prefer_longest")),
            microthemes_csv=str(raw.get("markers", {}).get("microthemes_csv", "")),
            remove_headings_from_analysis=bool(raw.get("remove_headings_from_analysis", True)),
            use_existing_segmentation=bool(raw.get("use_existing_segmentation", True)),
            auto_build_segmentation=bool(raw.get("auto_build_segmentation", False)),
            min_perspective_confidence_main=str(raw.get("min_perspective_confidence_main", "LOW")).upper(),
        )


def project_root_from_any(path: str | Path) -> Path:
    return Path(path).resolve().parent
