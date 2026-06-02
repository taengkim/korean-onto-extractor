from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from onto_extractor.lexicon import Lexicon

logger = logging.getLogger(__name__)

# POS tags considered "content nouns" for compound noun formation.
# NNB (의존명사: 것/수/때) is intentionally excluded — those are stopwords.
_COMPOUND_NOUN_TAGS: dict[str, frozenset[str]] = {
    "komoran":  frozenset({"NNG", "NNP"}),
    "mecab":    frozenset({"NNG", "NNP"}),
    "okt":      frozenset({"Noun"}),
    "hannanum": frozenset({"NC", "NQ"}),
    "kkma":     frozenset({"NNG", "NNP"}),
}
_DEFAULT_COMPOUND_TAGS: frozenset[str] = frozenset({"NNG", "NNP"})

_DEFAULT_SENT_ENDINGS: list[str] = ["다", "요", "죠", "군요"]
_MIN_SENT_LEN = 5


def _build_sent_split_re(endings: list[str]) -> re.Pattern[str]:
    """Build a sentence-splitting regex from a list of Korean terminal endings."""
    # Always split on punctuation sentence boundaries
    alts = [r"(?<=[.!?])\s+"]
    for ending in endings:
        escaped = re.escape(ending)
        alts.append(rf"(?<={escaped}\.)\s+")
    return re.compile("|".join(alts), re.UNICODE)


_SENT_SPLIT_RE = _build_sent_split_re(_DEFAULT_SENT_ENDINGS)


def split_sentences(
    text: str,
    endings: list[str] | None = None,
    min_len: int = _MIN_SENT_LEN,
) -> list[str]:
    """Split Korean text into sentences.

    Args:
        text: Raw body text (may contain newlines).
        endings: Korean terminal endings to split on (e.g. ["다", "요"]).
                 None uses the default set: 다/요/죠/군요.
        min_len: Minimum character length to keep a candidate sentence.

    Returns:
        List of non-empty sentence strings.
    """
    if endings is None:
        split_re = _SENT_SPLIT_RE
    else:
        split_re = _build_sent_split_re(endings)

    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n+", ". ", normalized)

    parts = split_re.split(normalized)
    sentences = [p.strip() for p in parts if len(p.strip()) >= min_len]
    logger.debug("split_sentences: %d sentences extracted", len(sentences))
    return sentences


@lru_cache(maxsize=2)
def _get_tagger(tagger_name: str) -> Any:
    """Lazily initialize and cache a KoNLPy tagger by name."""
    try:
        import konlpy.tag as kt
    except ImportError as exc:
        raise ImportError(
            "konlpy is required for morpheme analysis. Install it with: pip install konlpy"
        ) from exc

    tagger_map = {
        "komoran": kt.Komoran,
        "mecab": kt.Mecab,
        "okt": kt.Okt,
        "hannanum": kt.Hannanum,
        "kkma": kt.Kkma,
    }
    cls = tagger_map.get(tagger_name)
    if cls is None:
        raise ValueError(f"Unknown tagger {tagger_name!r}. Choose from: {list(tagger_map)}")

    try:
        instance = cls()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize KoNLPy tagger {tagger_name!r}. "
            f"Check that the required system binary is installed. Original error: {exc}"
        ) from exc

    logger.info("Initialized KoNLPy tagger: %s", tagger_name)
    return instance


def _collect_nouns_and_phrases(
    tagged: list[tuple[str, str]],
    compound_tags: frozenset[str],
    max_len: int = 2,
) -> list[str]:
    """Extract individual nouns and compound noun phrases from POS-tagged tokens.

    Scans the tagged token list for runs of consecutive content-noun morphemes.
    For each run it emits:
    - Every individual morpheme in the run
    - Every contiguous sub-sequence of length 2..max_len concatenated as a phrase

    Example (max_len=2):
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("이", "JX"), ("연구", "NNG")]
        compound_tags = frozenset({"NNG", "NNP"})
        → ["인공", "지능", "인공지능", "연구"]

    Example (max_len=3):
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("연구", "NNG")]
        → ["인공", "지능", "연구", "인공지능", "지능연구", "인공지능연구"]

    Args:
        tagged: List of (morpheme, pos_tag) tuples from tagger.pos().
        compound_tags: Set of POS tags treated as content nouns.
        max_len: Maximum n-gram length (default 2 = bigrams only).

    Returns:
        Deduplicated list preserving first-occurrence order.
    """
    seen: dict[str, None] = {}
    i = 0
    while i < len(tagged):
        word, pos = tagged[i]
        word = word.strip()
        if pos in compound_tags and word:
            # Collect consecutive noun run
            run: list[str] = [word]
            j = i + 1
            while j < len(tagged):
                w2, p2 = tagged[j]
                w2 = w2.strip()
                if p2 in compound_tags and w2:
                    run.append(w2)
                    j += 1
                else:
                    break

            # Emit individual morphemes
            for m in run:
                if m not in seen:
                    seen[m] = None

            # Emit n-grams for n in [2, max_len]
            for n in range(2, max_len + 1):
                for k in range(len(run) - n + 1):
                    phrase = "".join(run[k : k + n])
                    if phrase not in seen:
                        seen[phrase] = None

            i = j
        else:
            i += 1

    return list(seen.keys())


def extract_nouns(
    sentences: list[str],
    tagger: str = "komoran",
    use_phrases: bool = True,
    phrase_max_len: int = 2,
    compound_noun_tags: list[str] | None = None,
) -> list[list[str]]:
    """Extract nouns (and optionally compound noun phrases) from sentences.

    When *use_phrases* is True (default), the function uses ``tagger.pos()``
    to find consecutive content-noun morpheme runs and emits individual nouns
    plus compound noun phrases up to *phrase_max_len* morphemes long.
    When False it falls back to the simpler ``tagger.nouns()`` call.

    Args:
        sentences: List of sentence strings.
        tagger: KoNLPy tagger name ('komoran', 'mecab', 'okt', etc.).
        use_phrases: If True, also extract compound noun phrases.
        phrase_max_len: Maximum n-gram length for compound phrases (default 2 = bigrams).
        compound_noun_tags: Override POS tag set for compound noun formation.
            None = use tagger-specific defaults from _COMPOUND_NOUN_TAGS.

    Returns:
        Parallel list where each element is the list of nouns/phrases in that sentence.
    """
    t = _get_tagger(tagger)
    if compound_noun_tags is not None:
        compound_tags = frozenset(compound_noun_tags)
    else:
        compound_tags = _COMPOUND_NOUN_TAGS.get(tagger, _DEFAULT_COMPOUND_TAGS)
    result: list[list[str]] = []

    for sent in sentences:
        if use_phrases:
            try:
                tagged = t.pos(sent)
                nouns = _collect_nouns_and_phrases(tagged, compound_tags, max_len=phrase_max_len)
            except Exception as exc:
                logger.warning(
                    "pos() failed on sentence %r (%s) — falling back to nouns()",
                    sent[:40], exc,
                )
                try:
                    nouns = t.nouns(sent)
                except Exception as exc2:
                    logger.warning("nouns() fallback also failed: %s", exc2)
                    nouns = []
        else:
            try:
                nouns = t.nouns(sent)
            except Exception as exc:
                logger.warning("Tagger error on sentence %r: %s", sent[:40], exc)
                nouns = []

        result.append(nouns)

    logger.debug(
        "extract_nouns: processed %d sentences (use_phrases=%s, phrase_max_len=%d)",
        len(sentences), use_phrases, phrase_max_len,
    )
    return result


def normalize_nouns(
    noun_lists: list[list[str]],
    lexicon: "Lexicon",
) -> list[list[str]]:
    """Apply unit normalization and noise filtering to noun lists.

    Uses the lexicon's unit_map and noise_patterns to clean nouns extracted
    by the morpheme tagger before they reach term selection.

    Args:
        noun_lists: Per-sentence noun lists from extract_nouns.
        lexicon: Loaded Lexicon instance.

    Returns:
        Cleaned noun lists of the same length as the input.
    """
    result: list[list[str]] = []
    dropped = 0
    for nouns in noun_lists:
        cleaned: list[str] = []
        for noun in nouns:
            if lexicon.is_noise(noun):
                dropped += 1
                continue
            cleaned.append(lexicon.normalize(noun))
        result.append(cleaned)
    if dropped:
        logger.debug("normalize_nouns: dropped %d noisy tokens", dropped)
    return result
