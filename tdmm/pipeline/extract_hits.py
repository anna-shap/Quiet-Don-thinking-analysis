from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tdmm.utils.io_utils import read_text_with_fallback
from tdmm.utils.morph_utils import lemmatize_token
from tdmm.utils.storage_utils import load_df, save_df
from tdmm.utils.text_utils import iter_word_tokens, sentenize_text, split_headings_from_text


def _match_phrase_with_gap(lemmas: List[str], pattern: List[str], gap: int) -> List[Tuple[int, int]]:
    """Return inclusive token spans for lemma-pattern matches with a small allowed gap."""
    if not pattern:
        return []

    matches: List[Tuple[int, int]] = []
    n = len(lemmas)

    for i in range(n):
        if lemmas[i] != pattern[0]:
            continue
        pos = i
        ok = True
        for wanted in pattern[1:]:
            found = False
            upper = min(n, pos + gap + 2)
            for k in range(pos + 1, upper):
                if lemmas[k] == wanted:
                    pos = k
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            matches.append((i, pos))

    # Prefer wider spans when overlaps occur.
    matches = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    filtered: List[Tuple[int, int]] = []
    last_end = -1
    for start, end in matches:
        if start <= last_end:
            continue
        filtered.append((start, end))
        last_end = end
    return filtered




def _filter_phrase_candidates(
    candidates: List[Dict[str, Any]],
    *,
    mode: str = "prefer_longest",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve overlapping phrase candidates.

    ``prefer_longest`` keeps the widest phrase span inside a local overlap cluster.
    Dropped rows are still written to PHRASE_NESTED_CANDIDATES for diagnostics.
    """
    mode = (mode or "prefer_longest").lower().strip()
    if mode in {"keep_all", "all"} or not candidates:
        return candidates, []

    ordered = sorted(
        candidates,
        key=lambda r: (int(r["sent_id"]), int(r["token_idx"]), -(int(r["token_end_idx"]) - int(r["token_idx"])), str(r["marker_id"])),
    )
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    occupied: Dict[int, List[Tuple[int, int, str]]] = {}

    # Greedy by sentence and descending span length inside equal starts.
    ordered = sorted(
        ordered,
        key=lambda r: (int(r["sent_id"]), -(int(r["token_end_idx"]) - int(r["token_idx"])), int(r["token_idx"]), str(r["marker_id"])),
    )
    for r in ordered:
        sid = int(r["sent_id"]); start = int(r["token_idx"]); end = int(r["token_end_idx"])
        overlaps = [mid for s, e, mid in occupied.get(sid, []) if not (end < s or start > e)]
        if overlaps:
            rr = dict(r)
            rr["nested_overlap_with"] = ";".join(overlaps)
            dropped.append(rr)
            continue
        kept.append(r)
        occupied.setdefault(sid, []).append((start, end, str(r["marker_id"])))

    kept = sorted(kept, key=lambda r: (int(r["sent_id"]), int(r["token_idx"]), str(r["marker_id"])))
    dropped = sorted(dropped, key=lambda r: (int(r["sent_id"]), int(r["token_idx"]), str(r["marker_id"])))
    return kept, dropped

def _load_exclusions(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["lemma", "surface", "regex", "reason", "enabled"])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["lemma", "surface", "regex", "reason", "enabled"])
    df = pd.read_csv(p).fillna("")
    if "enabled" in df.columns:
        df = df[df["enabled"].astype(str).str.lower().isin(["1", "true", "yes", "да"])]
    return df


def _detect_noise(
    *,
    unit_norm: str,
    target_surface: str,
    sent_text: str,
    noise_filters: Dict[str, Any],
    exclusions: pd.DataFrame,
) -> Tuple[bool, str]:
    """Conservative surface-level false-positive filtering."""

    u = (unit_norm or "").lower().strip()
    surf = (target_surface or "").lower().strip()
    sent = sent_text or ""

    if noise_filters.get("podumayesh_particle", True):
        if u == "подумать" and surf == "подумаешь":
            return True, "PARTICLE_PODUMAYESH"

    for _, r in exclusions.iterrows():
        lemma = str(r.get("lemma", "")).lower().strip()
        surface = str(r.get("surface", "")).lower().strip()
        regex = str(r.get("regex", "")).strip()
        reason = str(r.get("reason", "")).strip() or "EXCLUSION_TABLE"

        if lemma and lemma != u:
            continue
        if surface and surface != surf:
            continue
        if regex:
            try:
                if not re.search(regex, sent, flags=re.IGNORECASE | re.UNICODE):
                    continue
            except re.error:
                continue
        return True, reason

    return False, ""


def _context_window(sents: Sequence[str], sent_id: int, radius: int) -> Tuple[str, str, str]:
    prev_text = sents[sent_id - 1] if sent_id > 0 else ""
    next_text = sents[sent_id + 1] if sent_id + 1 < len(sents) else ""
    start = max(0, sent_id - int(radius))
    end = min(len(sents), sent_id + int(radius) + 1)
    return prev_text, next_text, " ".join(sents[start:end])


def extract_corpus_hits(
    *,
    corpus_path: str | Path,
    markers_master_path: str | Path,
    out_dir: str | Path,
    encoding_fallbacks: Sequence[str],
    phrase_gap: int,
    max_hits_sample: int,
    max_sents: Optional[int] = None,
    noise_filters: Optional[Dict[str, Any]] = None,
    exclusions_csv: str | Path | None = None,
    context_window_sentences: int = 1,
    count_overlapping_word_hits: bool = False,
    remove_headings_from_analysis: bool = True,
    phrase_overlap_mode: str = "prefer_longest",
    export_csv: bool = False,
    csv_max_rows: int = 200_000,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    noise_filters = dict(noise_filters or {})
    exclusions = _load_exclusions(exclusions_csv)

    markers = load_df(markers_master_path)
    markers["unit_type"] = markers["unit_type"].astype(str).str.upper()

    word_markers = markers[markers["unit_type"] == "WORD"].copy()
    phrase_markers = markers[markers["unit_type"] == "PHRASE"].copy()

    word_lemmas = set(word_markers["unit_norm"].astype(str).tolist())
    lemma_to_ids: Dict[str, List[str]] = {}
    for _, r in word_markers.iterrows():
        lemma_to_ids.setdefault(str(r["unit_norm"]), []).append(str(r["marker_id"]))

    phrases: List[Tuple[str, List[str], str]] = []
    for _, r in phrase_markers.iterrows():
        pat = str(r["unit_norm"]).strip()
        if not pat:
            continue
        phrases.append((str(r["marker_id"]), pat.split(), pat))

    text, enc = read_text_with_fallback(corpus_path, encodings=encoding_fallbacks)
    text_for_analysis, heading_rows = split_headings_from_text(text, remove_headings=remove_headings_from_analysis)
    if heading_rows:
        save_df(pd.DataFrame(heading_rows), out / "HEADINGS", export_csv=True, csv_max_rows=csv_max_rows)
    else:
        save_df(pd.DataFrame(columns=["heading_id", "line_no", "char_pos_approx", "heading_type", "heading_text", "removed_from_analysis"]), out / "HEADINGS", export_csv=True, csv_max_rows=csv_max_rows)
    sents = [s.text for s in sentenize_text(text_for_analysis)]
    if max_sents is not None:
        sents = sents[: int(max_sents)]

    sent_rows: List[Dict[str, Any]] = []
    hit_rows: List[Dict[str, Any]] = []
    nested_phrase_rows: List[Dict[str, Any]] = []
    hit_id = 1

    for sent_id, sent_text in enumerate(sents):
        tokens = list(iter_word_tokens(sent_text))
        token_surfaces = [t.text for t in tokens]
        token_lemmas = [lemmatize_token(t.text)[0] for t in tokens]
        prev_text, next_text, ctx = _context_window(sents, sent_id, context_window_sentences)

        sent_rows.append(
            {
                "sent_id": sent_id,
                "sent_text": sent_text,
                "word_count": len(token_surfaces),
                "encoding": enc,
            }
        )

        phrase_candidates: List[Dict[str, Any]] = []
        if phrases:
            for marker_id, pattern, unit_norm in phrases:
                for start, end in _match_phrase_with_gap(token_lemmas, pattern, gap=phrase_gap):
                    phrase_surface = " ".join(token_surfaces[start : end + 1])
                    phrase_candidates.append(
                        {
                            "hit_type": "PHRASE",
                            "marker_id": marker_id,
                            "sent_id": sent_id,
                            "token_idx": start,
                            "token_end_idx": end,
                            "target_lemma": unit_norm,
                            "target_surface": phrase_surface,
                            "sent_text": sent_text,
                            "prev_sent_text": prev_text,
                            "next_sent_text": next_text,
                            "context_window": ctx,
                            "overlaps_phrase": False,
                            "phrase_overlap_ids": "",
                            "is_noise": False,
                            "noise_reason": "",
                            "count_in_main_stats": True,
                        }
                    )

        kept_phrases, dropped_phrases = _filter_phrase_candidates(phrase_candidates, mode=phrase_overlap_mode)
        phrase_spans: List[Tuple[int, int, str]] = []
        for row in kept_phrases:
            row = dict(row)
            row["hit_id"] = hit_id
            phrase_spans.append((int(row["token_idx"]), int(row["token_end_idx"]), str(row["marker_id"])))
            hit_rows.append(row)
            hit_id += 1
        for row in dropped_phrases:
            row = dict(row)
            row["candidate_id"] = len(nested_phrase_rows) + 1
            row["drop_reason"] = f"phrase_overlap_mode={phrase_overlap_mode}"
            nested_phrase_rows.append(row)

        # WORD hits. If a word is inside a PHRASE hit, keep it but mark overlap; this is
        # useful for auditing, while stats can still be filtered later if required.
        for idx, (surf, lem) in enumerate(zip(token_surfaces, token_lemmas)):
            if lem not in word_lemmas:
                continue
            overlap_ids = [mid for s, e, mid in phrase_spans if s <= idx <= e]
            for marker_id in lemma_to_ids.get(lem, []):
                is_noise, noise_reason = _detect_noise(
                    unit_norm=lem,
                    target_surface=surf,
                    sent_text=sent_text,
                    noise_filters=noise_filters,
                    exclusions=exclusions,
                )
                hit_rows.append(
                    {
                        "hit_id": hit_id,
                        "hit_type": "WORD",
                        "marker_id": marker_id,
                        "sent_id": sent_id,
                        "token_idx": idx,
                        "token_end_idx": idx,
                        "target_lemma": lem,
                        "target_surface": surf,
                        "sent_text": sent_text,
                        "prev_sent_text": prev_text,
                        "next_sent_text": next_text,
                        "context_window": ctx,
                        "overlaps_phrase": bool(overlap_ids),
                        "phrase_overlap_ids": ";".join(overlap_ids),
                        "is_noise": bool(is_noise),
                        "noise_reason": noise_reason,
                        "count_in_main_stats": bool((not overlap_ids) or count_overlapping_word_hits),
                    }
                )
                hit_id += 1

    sents_df = pd.DataFrame(sent_rows)
    hits_df = pd.DataFrame(hit_rows)

    save_df(sents_df, out / "SENTS", export_csv=export_csv, csv_max_rows=csv_max_rows)
    save_df(hits_df, out / "TD_HITS", export_csv=export_csv, csv_max_rows=csv_max_rows)
    save_df(pd.DataFrame(nested_phrase_rows), out / "PHRASE_NESTED_CANDIDATES", export_csv=True, csv_max_rows=csv_max_rows)

    if len(hits_df) > 0:
        sample = hits_df.head(int(max_hits_sample)).copy()
        save_df(sample, out / "TD_HITS_SAMPLE", export_csv=True, csv_max_rows=csv_max_rows)
