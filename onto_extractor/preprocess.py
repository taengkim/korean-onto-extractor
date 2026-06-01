from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from onto_extractor.lexicon import Lexicon

logger = logging.getLogger(__name__)

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


def extract_nouns(sentences: list[str], tagger: str = "komoran") -> list[list[str]]:
    """Extract nouns from each sentence using a KoNLPy tagger.

    Args:
        sentences: List of sentence strings.
        tagger: KoNLPy tagger name ('komoran', 'mecab', 'okt', etc.).

    Returns:
        Parallel list where each element is the list of nouns in that sentence.
    """
    t = _get_tagger(tagger)
    result: list[list[str]] = []
    for sent in sentences:
        try:
            nouns = t.nouns(sent)
        except Exception as exc:
            logger.warning("Tagger error on sentence %r: %s", sent[:40], exc)
            nouns = []
        result.append(nouns)
    logger.debug("extract_nouns: processed %d sentences", len(sentences))
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
