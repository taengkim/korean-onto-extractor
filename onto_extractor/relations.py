from __future__ import annotations

import logging
import re
from collections import Counter
from itertools import combinations

logger = logging.getLogger(__name__)


def _find_concept(raw: str, concepts: set[str]) -> str | None:
    """Find the longest concept that is a prefix of raw.

    Korean particles (조사) are syllable blocks in [가-힣], so regex groups
    may capture "사과가" when only "사과" is the concept. This function
    strips trailing particles by finding the longest known concept prefix.
    """
    if raw in concepts:
        return raw
    matches = [c for c in concepts if raw.startswith(c)]
    return max(matches, key=len) if matches else None


# ---------------------------------------------------------------------------
# Korean Hearst patterns
#
# Each entry: (name, compiled_pattern, hyper_first)
#   hyper_first=True  means the grammatical order in text is hyper before hypo
#   hyper_first=False means hypo appears first
#
# Named groups: (?P<hypo>...) and (?P<hyper>...)
# ---------------------------------------------------------------------------

_RAW_PATTERNS: list[tuple[str, str, bool]] = [
    # 1. "X는 Y의 일종이다" → X is-a Y
    (
        "일종이다",
        r"(?P<hypo>[가-힣]+)(?:는|은)\s+(?P<hyper>[가-힣]+)의\s+일종이",
        False,
    ),
    # 2. "X와/과 같은 Y" or "X 같은 Y" → X is-a Y
    (
        "같은",
        r"(?P<hypo>[가-힣]+)(?:와|과|)\s*같은\s+(?P<hyper>[가-힣]+)",
        False,
    ),
    # 3. "Y에는 X가 포함된다/속한다" → X is-a Y
    (
        "포함된다",
        r"(?P<hyper>[가-힣]+)에(?:는|)\s+(?P<hypo>[가-힣]+)가?\s+(?:포함된|속한)",
        True,
    ),
    # 4. "X는 Y이다/이에요/입니다/이란" → X is-a Y
    (
        "이다",
        r"(?P<hypo>[가-힣]+)(?:는|은|이)\s+(?P<hyper>[가-힣]+)(?:이다|이에요|입니다|이란)",
        False,
    ),
    # 5. "X, Y 등 Z" or "X 등의 Z" → (last item before 등) is-a Z
    # The non-capturing prefix skips earlier list items, leaving hypo as the final one.
    (
        "등",
        r"(?:[가-힣]+\s*,\s*)*(?P<hypo>[가-힣]+)\s+등(?:의|은|는|)\s+(?P<hyper>[가-힣]+)",
        False,
    ),
    # 6. "X(이)라고 불리는 Y" → X is-a Y
    (
        "불리는",
        r"(?P<hypo>[가-힣]+)(?:이라고|라고)\s+불리는\s+(?P<hyper>[가-힣]+)",
        False,
    ),
    # 7. "Y의 하나인 X" → X is-a Y
    (
        "하나인",
        r"(?P<hyper>[가-힣]+)의\s+하나인\s+(?P<hypo>[가-힣]+)",
        True,
    ),
    # 8. "Y의 종류인 X" → X is-a Y
    (
        "종류인",
        r"(?P<hyper>[가-힣]+)의\s+종류인\s+(?P<hypo>[가-힣]+)",
        True,
    ),
    # 9. "X 등을 포함한 Y" → X is-a Y
    (
        "포함한",
        r"(?P<hypo>[가-힣]+)\s+등을\s+포함한\s+(?P<hyper>[가-힣]+)",
        False,
    ),
    # 10. "X는 Y의 한 종류" → X is-a Y
    (
        "한종류",
        r"(?P<hypo>[가-힣]+)(?:는|은)\s+(?P<hyper>[가-힣]+)의\s+한\s+종류",
        False,
    ),
]

_COMPILED_PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    (name, re.compile(pattern, re.UNICODE), hyper_first)
    for name, pattern, hyper_first in _RAW_PATTERNS
]


def extract_isa(
    sentences: list[str],
    concepts: set[str],
) -> list[tuple[str, str]]:
    """Extract is-a (hyponym, hypernym) pairs using Korean Hearst patterns.

    Only pairs where both terms are in concepts are emitted.
    Duplicates are removed while preserving first-occurrence order.

    Args:
        sentences: List of Korean sentences.
        concepts: Set of concept strings from select_terms.

    Returns:
        List of (hyponym, hypernym) tuples.
    """
    seen: dict[tuple[str, str], None] = {}

    for sent in sentences:
        for pattern_name, pattern, hyper_first in _COMPILED_PATTERNS:
            for match in pattern.finditer(sent):
                raw_hypo = match.group("hypo").strip()
                raw_hyper = match.group("hyper").strip()
                hypo = _find_concept(raw_hypo, concepts)
                hyper = _find_concept(raw_hyper, concepts)
                if hypo and hyper and hypo != hyper:
                    pair = (hypo, hyper)
                    if pair not in seen:
                        seen[pair] = None
                        logger.debug("is-a [%s]: (%s, %s)", pattern_name, hypo, hyper)

    result = list(seen.keys())
    logger.info("extract_isa: found %d is-a pairs", len(result))
    return result


def extract_cooccurrence(
    noun_lists: list[list[str]],
    concepts: set[str],
    min_count: int,
) -> list[tuple[str, str]]:
    """Extract co-occurrence-based related_to pairs.

    Two concepts co-occur when they both appear in the same sentence.
    Pairs with count >= min_count are returned sorted for determinism.

    Args:
        noun_lists: Per-sentence noun lists from extract_nouns.
        concepts: Set of selected concept strings.
        min_count: Minimum co-occurrence count to include a pair.

    Returns:
        List of (concept_a, concept_b) tuples in canonical (sorted) order.
    """
    cooc: Counter[tuple[str, str]] = Counter()

    for nouns in noun_lists:
        sent_concepts = sorted({n for n in nouns if n in concepts})
        for a, b in combinations(sent_concepts, 2):
            pair = (a, b) if a <= b else (b, a)
            cooc[pair] += 1

    result = [
        pair
        for pair, count in sorted(cooc.items())
        if count >= min_count
    ]
    logger.info("extract_cooccurrence: found %d co-occurrence pairs", len(result))
    return result
