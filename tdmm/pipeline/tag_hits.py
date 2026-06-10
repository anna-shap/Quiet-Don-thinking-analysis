from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tdmm.utils.morph_utils import get_morph
from tdmm.utils.storage_utils import load_df, save_df
from tdmm.utils.text_utils import contains_any, extract_proper_names, sentence_window


PERSPECTIVE_LABELS_RU: Dict[str, str] = {
    "HERO_INNER_SPEECH": "герой: внутренняя речь",
    "HERO_DIRECT_SPEECH": "герой: прямая речь",
    "NARRATOR_ABOUT_HERO": "повествователь о герое",
    "OTHERS_TO_HERO": "другие персонажи обращаются к герою",
    "OTHERS_ABOUT_HERO": "другие персонажи говорят о герое",
    "OTHERS_NEAR_HERO": "другие персонажи в контексте рядом с героем",
    "UNKNOWN": "неопределено",
}

OTHERS_SPEECHACT_LABELS_RU: Dict[str, str] = {
    "PROMPT": "побуждение (совет/призыв думать)",
    "EVALUATE": "оценка (критика/похвала способности думать)",
    "STATEMENT": "констатация",
    "UNKNOWN": "неопределено",
}

# Speech and inner-speech verbs used for shallow attribution. This is a
# transparent heuristic, not a full parser.
_SPEAKER_VERBS = r"(сказал|сказала|ответил|ответила|спросил|спросила|крикнул|крикнула|шепнул|шепнула|прошептал|прошептала|буркнул|буркнула|промолвил|промолвила|молвил|молвила|воскликнул|воскликнула|произнес|произнесла|произнёс|произнесла|проговорил|проговорила|заметил|заметила|возразил|возразила|перебил|перебила|подумал|подумала|решил|решила|вспомнил|вспомнила|мысленно\s+сказал|мысленно\s+сказала|мысленно\s+проговорил|мысленно\s+проговорила)"

_HERO_PRONOUN_RE = re.compile(r"\b(он|ему|его|им|нем|нём|сам)\b", re.IGNORECASE | re.UNICODE)
_DIRECT_SPEECH_RE = re.compile(r"(^\s*[—-]\s+)|([«\"].+[»\"])", re.UNICODE)
_EVAL_PATTERNS = [
    re.compile(r"\bне\s+(дума(ет|ют|ешь)|сообража(ет|ют|ешь)|понима(ет|ют|ешь))\b", re.IGNORECASE),
    re.compile(r"\bбез\s+(голов(ы|а)|мозг(ов|а))\b", re.IGNORECASE),
    re.compile(r"\bдурн(ой|ая|ые|ым)\b", re.IGNORECASE),
]


def _norm_variants(name_variants: Sequence[str]) -> List[str]:
    base = [str(v).strip() for v in name_variants if str(v).strip()]
    extra = [
        "Григорий", "Григория", "Григорию", "Григорием", "Григории",
        "Гриша", "Гришу", "Грише", "Гришей", "Гришка", "Гришку", "Гришке", "Гришкой",
        "Мелехов", "Мелехова", "Мелехову", "Мелеховым", "Мелехове",
    ]
    out: List[str] = []
    for x in base + extra:
        if x not in out:
            out.append(x)
    return out


def _find_name_variant(text: str, name_variants: Sequence[str]) -> str:
    for v in name_variants:
        if re.search(r"(?<![А-Яа-яЁё])" + re.escape(v) + r"(?![А-Яа-яЁё])", text, flags=re.IGNORECASE | re.UNICODE):
            return str(v)
    return ""


def _speaker_tag(sentence: str, name_variants: Sequence[str]) -> Tuple[str, str, str]:
    """Shallow speaker attribution for dialogue-like sentences.

    Returns (role, evidence, detected_name). The function intentionally stays
    transparent: it uses only surface patterns that can be explained in the
    thesis documentation.
    """
    t = sentence or ""

    for v in name_variants:
        # "— ... — сказал Григорий" or "Григорий сказал: ..."
        if re.search(_SPEAKER_VERBS + r"\s+" + re.escape(v) + r"\b", t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"speech_verb_after_quote:{v}", v
        if re.search(re.escape(v) + r"\s+" + _SPEAKER_VERBS, t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"speech_verb_after_name:{v}", v

    m = re.search(_SPEAKER_VERBS + r"\s+([А-ЯЁ][а-яё]+)\b", t, flags=re.UNICODE)
    if m:
        name = m.group(2)
        if not _find_name_variant(name, name_variants):
            return "OTHER", f"speech_verb_other:{name}", name

    m = re.search(r"([А-ЯЁ][а-яё]+)\s+" + _SPEAKER_VERBS, t, flags=re.UNICODE)
    if m:
        name = m.group(1)
        if not _find_name_variant(name, name_variants):
            return "OTHER", f"other_name_before_speech_verb:{name}", name

    return "UNKNOWN", "speaker_not_detected", ""


_HERO_DATIVE_FORMS = (
    "Григорию", "Грише", "Гришке", "Мелехову",
    "Григория", "Гришу", "Гришку", "Мелехова",
)


def _addressee_tag(sentence: str, name_variants: Sequence[str]) -> Tuple[str, str, str]:
    """Detect whether the utterance is addressed to the hero.

    This is deliberately conservative. It looks for dative/accusative hero
    forms near speech verbs or for direct vocative-like commands.
    """
    t = sentence or ""
    dative = list(_HERO_DATIVE_FORMS)
    for v in dative:
        if re.search(_SPEAKER_VERBS + r".{0,35}(?:к\s+)?" + re.escape(v) + r"\b", t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"speech_verb_to_hero:{v}", v
        if re.search(r"(?:к\s+)?" + re.escape(v) + r".{0,35}" + _SPEAKER_VERBS, t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"hero_dative_near_speech_verb:{v}", v

    # Direct address inside speech: "Григорий, подумай", "Подумай, Гриша".
    for v in name_variants:
        if re.search(re.escape(v) + r"[,!]?\s+(подумай|думай|пойми|сообрази|вспомни|решай)\b", t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"vocative_hero_before_prompt:{v}", v
        if re.search(r"\b(подумай|думай|пойми|сообрази|вспомни|решай)[,!]?\s+" + re.escape(v) + r"\b", t, flags=re.IGNORECASE | re.UNICODE):
            return "HERO", f"vocative_hero_after_prompt:{v}", v

    if _find_name_variant(t, name_variants):
        return "HERO_PROBABLE", "hero_name_in_utterance", _find_name_variant(t, name_variants)
    return "UNKNOWN", "addressee_not_detected", ""

def _is_direct_speech(sent_text: str) -> bool:
    t = (sent_text or "").strip()
    if not t:
        return False
    return bool(_DIRECT_SPEECH_RE.search(t))


def _is_inner_speech(sent_text: str) -> bool:
    return bool(re.search(r"\b(мысленно|про\s+себя)\b", sent_text or "", flags=re.IGNORECASE | re.UNICODE))


def _is_imperative(surface: str) -> bool:
    surf = surface.strip()
    if not surf:
        return False
    try:
        m = get_morph()
        for p in m.parse(surf):
            if "impr" in p.tag:
                return True
    except Exception:
        pass
    return False


def _others_speechact(sent_text: str, target_surface: str) -> str:
    if _is_imperative(target_surface):
        return "PROMPT"
    for pat in _EVAL_PATTERNS:
        if pat.search(sent_text):
            return "EVALUATE"
    if re.search(r"\b(подумай|пойми|сообрази|вспомни|решай|думай)\b", sent_text, flags=re.IGNORECASE | re.UNICODE):
        return "PROMPT"
    return "STATEMENT"


def _assign_segments(
    hits: pd.DataFrame,
    seg_csv: str | Path,
    *,
    id_col: str,
    name_col: str,
    out_id_col: str,
    out_name_col: str,
) -> None:
    seg = pd.read_csv(seg_csv)

    if id_col not in seg.columns:
        if id_col == "book_id" and "part_id" in seg.columns:
            seg = seg.rename(columns={"part_id": "book_id"})
        if id_col == "part_id" and "book_id" in seg.columns and "part_id" not in seg.columns:
            seg = seg.rename(columns={"book_id": "part_id"})
    if name_col not in seg.columns:
        if name_col == "book_name" and "book" in seg.columns:
            seg = seg.rename(columns={"book": "book_name"})
        if name_col == "part_name" and "part" in seg.columns:
            seg = seg.rename(columns={"part": "part_name"})

    seg["start_sent"] = seg["start_sent"].astype(int)
    seg["end_sent"] = seg["end_sent"].astype(int)
    seg[id_col] = seg[id_col].astype(int)
    seg[name_col] = seg[name_col].astype(str)

    intervals = pd.IntervalIndex.from_arrays(seg["start_sent"], seg["end_sent"], closed="both")
    sids = hits["sent_id"].astype(int).to_numpy()
    idx = intervals.get_indexer(sids)

    hits[out_id_col] = [int(seg.iloc[i][id_col]) if i != -1 else -1 for i in idx]
    hits[out_name_col] = [str(seg.iloc[i][name_col]) if i != -1 else "UNKNOWN" for i in idx]


def tag_hits(
    *,
    out_dir: str | Path,
    name_variants: Sequence[str],
    window_sentences: int,
    parts_csv: Optional[str | Path] = None,
    books_csv: Optional[str | Path] = None,
    max_hits_sample: int = 500,
    export_csv: bool = False,
    csv_max_rows: int = 200_000,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = _norm_variants(name_variants)
    sents = load_df(out_dir / "SENTS").copy()
    hits = load_df(out_dir / "TD_HITS").copy()
    try:
        markers_meta = load_df(out_dir / "MARKERS_MASTER").copy()
        markers_meta["marker_id"] = markers_meta["marker_id"].astype(str)
        hits["marker_id"] = hits["marker_id"].astype(str)
        meta_cols = [c for c in ["marker_id", "marker_label", "microtheme", "phrase_family", "unit_type", "marker_source", "precision_level", "source_type", "risk_comment", "unit_raw", "unit_norm"] if c in markers_meta.columns]
        hits = hits.merge(markers_meta[meta_cols], on="marker_id", how="left")
    except Exception:
        pass

    sents["sent_id"] = sents["sent_id"].astype(int)
    sents["sent_text"] = sents["sent_text"].astype(str)
    sents["is_direct_speech"] = sents["sent_text"].apply(_is_direct_speech)
    sents["has_inner_speech"] = sents["sent_text"].apply(_is_inner_speech)
    sents["has_hero_pronoun"] = sents["sent_text"].apply(lambda t: bool(_HERO_PRONOUN_RE.search(str(t))))
    sents["hero_name_variant"] = sents["sent_text"].apply(lambda t: _find_name_variant(str(t), names))
    sents["has_hero_name"] = sents["hero_name_variant"].astype(str).str.len() > 0

    total = len(sents)
    has_hero_near: List[bool] = []
    hero_near_evidence: List[str] = []
    for idx in range(total):
        w = sentence_window(idx, total, int(window_sentences))
        near_slice = sents.loc[w.start : w.end]
        rows = near_slice[near_slice["has_hero_name"]]
        has_hero_near.append(bool(len(rows)))
        if len(rows):
            first = rows.iloc[0]
            rel = int(first["sent_id"]) - int(sents.iloc[idx]["sent_id"])
            loc = "same_sentence" if rel == 0 else ("prev_sentence" if rel < 0 else "next_sentence")
            pron = "+pronoun" if bool(sents.iloc[idx].get("has_hero_pronoun", False)) and rel != 0 else ""
            hero_near_evidence.append(f"{loc}:{first['hero_name_variant']}{pron}")
        else:
            hero_near_evidence.append("")

    sents["has_hero_near"] = has_hero_near
    sents["hero_near_evidence"] = hero_near_evidence

    sent_map = sents.set_index("sent_id")[[
        "sent_text", "is_direct_speech", "has_inner_speech", "has_hero_pronoun", "has_hero_name",
        "hero_name_variant", "has_hero_near", "hero_near_evidence"
    ]]

    def link_label(sid: int) -> str:
        r = sent_map.loc[sid]
        if bool(r["has_hero_name"]):
            return "YES"
        if bool(r["has_hero_near"]):
            return "PROBABLE"
        return "NO"

    def link_evidence(sid: int) -> str:
        r = sent_map.loc[sid]
        if bool(r["has_hero_name"]):
            return f"same_sentence:{r['hero_name_variant']}"
        return str(r["hero_near_evidence"] or "")

    def perspective_info(sid: int, target_surface: str = "") -> Tuple[str, str, str, str, str]:
        r = sent_map.loc[sid]
        sent_text = str(r["sent_text"])
        link = link_label(sid)
        speaker, speaker_ev, speaker_name = _speaker_tag(sent_text, names)
        addressee, addressee_ev, addressee_name = _addressee_tag(sent_text, names)
        speechact = _others_speechact(sent_text, target_surface or "")

        if bool(r["is_direct_speech"]):
            if speaker == "HERO":
                if bool(r["has_inner_speech"]):
                    return "HERO_INNER_SPEECH", f"direct_speech+inner_marker;{speaker_ev}", "HIGH", speaker_name or "HERO", ""
                return "HERO_DIRECT_SPEECH", f"direct_speech;{speaker_ev}", "HIGH", speaker_name or "HERO", ""

            if speaker == "OTHER":
                if addressee in {"HERO", "HERO_PROBABLE"}:
                    conf = "HIGH" if addressee == "HERO" else "MEDIUM"
                    return "OTHERS_TO_HERO", f"direct_speech;{speaker_ev};{addressee_ev};speechact={speechact}", conf, speaker_name or "OTHER", addressee_name or "HERO"
                if link in {"YES", "PROBABLE"} and speechact == "PROMPT":
                    conf = "MEDIUM" if link == "YES" else "LOW"
                    return "OTHERS_TO_HERO", f"direct_speech;{speaker_ev};speechact=PROMPT;hero_link={link}", conf, speaker_name or "OTHER", "HERO_PROBABLE"
                if link == "YES":
                    return "OTHERS_ABOUT_HERO", f"direct_speech;{speaker_ev};hero_same_sentence", "MEDIUM", speaker_name or "OTHER", ""
                if link == "PROBABLE":
                    return "OTHERS_NEAR_HERO", f"direct_speech;{speaker_ev};hero_near", "LOW", speaker_name or "OTHER", ""
                return "UNKNOWN", f"direct_speech;{speaker_ev};no_hero_link", "LOW", speaker_name or "OTHER", ""

            if link == "YES" and bool(r["has_inner_speech"]):
                return "HERO_INNER_SPEECH", "direct_or_inner_speech;hero_same_sentence;inner_marker", "MEDIUM", "UNKNOWN", ""
            if addressee in {"HERO", "HERO_PROBABLE"}:
                conf = "MEDIUM" if addressee == "HERO" else "LOW"
                return "OTHERS_TO_HERO", f"direct_speech;unknown_speaker;{addressee_ev};speechact={speechact}", conf, "UNKNOWN", addressee_name or "HERO"
            if link in {"YES", "PROBABLE"} and speechact == "PROMPT":
                return "OTHERS_TO_HERO", f"direct_speech;unknown_speaker;speechact=PROMPT;hero_link={link}", "LOW", "UNKNOWN", "HERO_PROBABLE"
            if link == "YES":
                return "OTHERS_ABOUT_HERO", "direct_speech;unknown_speaker;hero_same_sentence", "LOW", "UNKNOWN", ""
            if link == "PROBABLE":
                return "OTHERS_NEAR_HERO", "direct_speech;unknown_speaker;hero_near", "LOW", "UNKNOWN", ""
            return "UNKNOWN", "direct_speech;speaker_not_detected;no_hero_link", "LOW", "UNKNOWN", ""

        if speaker == "OTHER" and addressee in {"HERO", "HERO_PROBABLE"} and speechact == "PROMPT":
            conf = "HIGH" if addressee == "HERO" else "MEDIUM"
            return "OTHERS_TO_HERO", f"reported_speech;{speaker_ev};{addressee_ev};speechact=PROMPT", conf, speaker_name or "OTHER", addressee_name or "HERO"

        if bool(r["has_inner_speech"]) and link in {"YES", "PROBABLE"}:
            conf = "HIGH" if link == "YES" else "MEDIUM"
            return "HERO_INNER_SPEECH", f"inner_speech_marker;hero_link={link}", conf, "HERO_PROBABLE", ""

        if link in {"YES", "PROBABLE"}:
            conf = "HIGH" if link == "YES" else "MEDIUM"
            if bool(r.get("has_hero_pronoun", False)) and link == "PROBABLE":
                return "NARRATOR_ABOUT_HERO", "narrative_context;hero_near+pronoun", conf, "NARRATOR", ""
            return "NARRATOR_ABOUT_HERO", f"narrative_context;hero_link={link}", conf, "NARRATOR", ""

        return "UNKNOWN", "no_direct_speech;no_hero_link", "LOW", "UNKNOWN", ""

    hits["MelekhovLink"] = hits["sent_id"].astype(int).apply(link_label)
    hits["MelekhovEvidence"] = hits["sent_id"].astype(int).apply(link_evidence)

    infos = hits.apply(lambda r: perspective_info(int(r["sent_id"]), str(r.get("target_surface", ""))), axis=1)
    hits["Perspective"] = [x[0] for x in infos]
    hits["PerspectiveEvidence"] = [x[1] for x in infos]
    hits["PerspectiveConfidence"] = [x[2] for x in infos]
    hits["SpeakerDetected"] = [x[3] for x in infos]
    hits["AddresseeDetected"] = [x[4] for x in infos]

    hits["OthersSpeechAct"] = ""
    mask_others = hits["Perspective"].isin(["OTHERS_TO_HERO", "OTHERS_ABOUT_HERO", "OTHERS_NEAR_HERO"])
    if mask_others.any():
        hits.loc[mask_others, "OthersSpeechAct"] = hits.loc[mask_others].apply(
            lambda r: _others_speechact(str(r.get("sent_text", "")), str(r.get("target_surface", ""))),
            axis=1,
        )

    if books_csv is not None and Path(books_csv).exists():
        _assign_segments(hits, books_csv, id_col="book_id", name_col="book_name", out_id_col="BookId", out_name_col="Book")
    else:
        hits["BookId"] = -1
        hits["Book"] = "UNKNOWN"

    if parts_csv is not None and Path(parts_csv).exists():
        _assign_segments(hits, parts_csv, id_col="part_id", name_col="part_name", out_id_col="PartId", out_name_col="Part")
    else:
        hits["PartId"] = -1
        hits["Part"] = "UNKNOWN"

    save_df(hits, out_dir / "TD_HITS_TAGGED", export_csv=export_csv, csv_max_rows=csv_max_rows)

    n = int(max_hits_sample) if max_hits_sample else 0
    if n and len(hits) > 0:
        priority = pd.Categorical(hits["MelekhovLink"], categories=["YES", "PROBABLE", "NO"], ordered=True)
        sample = hits.assign(_priority=priority).sort_values(["_priority", "sent_id", "token_idx"]).drop(columns=["_priority"])
        sample = sample.head(min(n, len(sample))).copy()
        sample.to_csv(out_dir / "TD_HITS_TAGGED_SAMPLE.csv", index=False, encoding="utf-8-sig")
