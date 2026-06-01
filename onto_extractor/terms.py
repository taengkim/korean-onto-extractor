from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer

from onto_extractor.config import ExtractConfig

if TYPE_CHECKING:
    from onto_extractor.lexicon import Lexicon

logger = logging.getLogger(__name__)


def select_terms(
    noun_lists: list[list[str]],
    cfg: ExtractConfig,
    lexicon: "Lexicon | None" = None,
) -> list[str]:
    """Select the most representative concept terms from noun lists.

    Steps:
    1. Flatten all nouns and count global frequency.
    2. Filter by min_freq, min_term_len, and stopwords (via lexicon).
    3. Compute TF-IDF across sentences (each sentence = a "document").
    4. Inject known entities (from lexicon) that meet min_freq.
    5. Return top_terms by descending mean TF-IDF score.

    Args:
        noun_lists: Per-sentence lists of nouns (from extract_nouns / normalize_nouns).
        cfg: Configuration object with filter thresholds.
        lexicon: Optional Lexicon for stopword filtering and entity boosting.

    Returns:
        Ordered list of selected concept strings (highest TF-IDF first).
    """
    if not noun_lists or all(len(ns) == 0 for ns in noun_lists):
        logger.warning("select_terms: no nouns provided, returning empty list")
        return []

    global_freq: Counter[str] = Counter()
    for nouns in noun_lists:
        global_freq.update(nouns)

    # Step 2: frequency + length filter + stopword filter
    candidates: set[str] = {
        term
        for term, freq in global_freq.items()
        if freq >= cfg.min_freq
        and len(term) >= cfg.min_term_len
        and (lexicon is None or not lexicon.is_stopword(term))
    }

    if not candidates:
        logger.warning(
            "select_terms: no candidates after frequency/length/stopword filter "
            "(min_freq=%d, min_term_len=%d)",
            cfg.min_freq,
            cfg.min_term_len,
        )
        return []

    # Step 4: inject known entities that appear in text (even below top_terms cut)
    if lexicon:
        entity_additions = {
            entity
            for entity in lexicon._entity_set
            if global_freq.get(entity, 0) >= cfg.min_freq
            and len(entity) >= cfg.min_term_len
        }
        before = len(candidates)
        candidates |= entity_additions
        added = len(candidates) - before
        if added:
            logger.debug(
                "select_terms: injected %d entities from entity dictionary", added
            )

    vocab_sorted = sorted(candidates)

    docs = [
        " ".join(n for n in nouns if n in candidates)
        for nouns in noun_lists
    ]
    non_empty_docs = [d for d in docs if d.strip()]

    if not non_empty_docs:
        logger.warning("select_terms: all sentence documents are empty after candidate filtering")
        return vocab_sorted[: cfg.top_terms]

    vectorizer = TfidfVectorizer(
        vocabulary=vocab_sorted,
        token_pattern=r"(?u)\b\S+\b",
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(non_empty_docs)
    mean_scores = tfidf_matrix.mean(axis=0).A1

    scored = sorted(
        zip(vocab_sorted, mean_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    selected = [term for term, _score in scored[: cfg.top_terms]]
    logger.info("select_terms: selected %d / %d candidates", len(selected), len(candidates))
    return selected
