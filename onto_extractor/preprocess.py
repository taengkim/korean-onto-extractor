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

_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=죠\.)\s+|(?<=군요\.)\s+",
    re.UNICODE,
)

_MIN_SENT_LEN = 5


def split_sentences(text: str) -> list[str]:
    """Split Korean text into sentences.

    Args:
        text: Raw body text (may contain newlines).

    Returns:
        List of non-empty sentence strings.
    """
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n+", ". ", normalized)

    parts = _SENT_SPLIT_RE.split(normalized)
    sentences = [p.strip() for p in parts if len(p.strip()) >= _MIN_SENT_LEN]
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
) -> list[str]:
    """Extract individual nouns and bigram compound noun phrases from POS-tagged tokens.

    Scans the tagged token list for runs of consecutive content-noun morphemes.
    For each run it emits:
    - Every individual morpheme in the run
    - Every adjacent pair (bigram) concatenated as a compound noun

    Example:
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("이", "JX"), ("연구", "NNG")]
        compound_tags = frozenset({"NNG", "NNP"})
        → ["인공", "지능", "인공지능", "연구"]

    Args:
        tagged: List of (morpheme, pos_tag) tuples from tagger.pos().
        compound_tags: Set of POS tags treated as content nouns.

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

            # Emit bigrams (adjacent pairs)
            for k in range(len(run) - 1):
                phrase = run[k] + run[k + 1]
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
) -> list[list[str]]:
    """Extract nouns (and optionally compound noun phrases) from sentences.

    When *use_phrases* is True (default), the function uses ``tagger.pos()``
    to find consecutive content-noun morpheme runs and emits both individual
    nouns and bigram compound nouns (복합명사).  When False it falls back to
    the simpler ``tagger.nouns()`` call (original behaviour).

    Args:
        sentences: List of sentence strings.
        tagger: KoNLPy tagger name ('komoran', 'mecab', 'okt', etc.).
        use_phrases: If True, also extract bigram compound noun phrases.

    Returns:
        Parallel list where each element is the list of nouns/phrases in that sentence.
    """
    t = _get_tagger(tagger)
    compound_tags = _COMPOUND_NOUN_TAGS.get(tagger, _DEFAULT_COMPOUND_TAGS)
    result: list[list[str]] = []

    for sent in sentences:
        if use_phrases:
            try:
                tagged = t.pos(sent)
                nouns = _collect_nouns_and_phrases(tagged, compound_tags)
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
        "extract_nouns: processed %d sentences (use_phrases=%s)",
        len(sentences), use_phrases,
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
