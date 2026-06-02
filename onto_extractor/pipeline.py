from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from onto_extractor.config import ExtractConfig
from onto_extractor.fetch import fetch_text
from onto_extractor.lexicon import Lexicon
from onto_extractor.owl_builder import build_owl
from onto_extractor.preprocess import extract_nouns, normalize_nouns, split_sentences
from onto_extractor.relations import extract_cooccurrence, extract_isa, extract_typed_relations
from onto_extractor.terms import select_terms

logger = logging.getLogger(__name__)


def _url_to_filename(url: str) -> str:
    """Derive a deterministic filename fragment from a URL."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"onto_{digest}.owl"


def run_one(url: str, cfg: ExtractConfig) -> Path:
    """Run the full extraction pipeline for a single URL.

    Pipeline stages:
    1. fetch       → raw text
    2. preprocess  → sentences + raw noun lists
    3. normalize   → noise filtering + unit normalization (via lexicon)
    4. terms       → concept candidates (stopword filtering + entity injection via lexicon)
    5. relations   → is-a pairs (Hearst) + typed pairs (relation triggers) + co-occurrence
    6. owl_builder → .owl file

    Args:
        url: Web page URL to process.
        cfg: Configuration controlling all pipeline parameters.

    Returns:
        Path to the written .owl file.
    """
    logger.info("=== Processing URL: %s ===", url)

    # Load lexicon (uses bundled lexicons/ when cfg.lexicon_dir is None)
    lexicon = Lexicon.load(cfg.lexicon_dir)

    # Stage 1: Fetch
    text = fetch_text(url)

    # Stage 2: Preprocess
    sentences = split_sentences(text, endings=cfg.sent_endings, min_len=cfg.min_sent_len)
    if not sentences:
        raise ValueError(f"No sentences extracted from {url!r}")
    noun_lists = extract_nouns(
        sentences,
        tagger=cfg.tagger,
        use_phrases=cfg.use_phrases,
        phrase_max_len=cfg.phrase_max_len,
        compound_noun_tags=cfg.compound_noun_tags,
    )

    # Stage 3: Normalize (unit normalization + noise filtering)
    noun_lists = normalize_nouns(noun_lists, lexicon)

    # Stage 4: Terms (stopword filtering + entity injection)
    concepts = select_terms(noun_lists, cfg, lexicon=lexicon)
    if not concepts:
        raise ValueError(
            f"No concept terms selected for {url!r} — "
            "consider lowering min_freq or min_term_len"
        )
    concept_set = set(concepts)

    # Stage 5: Relations
    isa_pairs = extract_isa(sentences, concept_set)
    typed_pairs = extract_typed_relations(sentences, concept_set, lexicon)

    cooc_pairs: list[tuple[str, str]] = []
    if cfg.use_cooccurrence:
        cooc_pairs = extract_cooccurrence(noun_lists, concept_set, cfg.cooc_min_count)

    # Stage 6: OWL output
    out_path = Path(cfg.output_dir) / _url_to_filename(url)
    build_owl(concepts, isa_pairs, cooc_pairs, cfg, out_path, typed_pairs=typed_pairs)

    logger.info("=== Done: %s ===", out_path)
    return out_path


def run_all(cfg: ExtractConfig) -> list[Path]:
    """Process all URLs listed in cfg.urls.

    Failures on individual URLs are logged and skipped.

    Args:
        cfg: Configuration with a populated urls list.

    Returns:
        List of Paths for successfully written .owl files.
    """
    if not cfg.urls:
        logger.warning("run_all: cfg.urls is empty — nothing to process")
        return []

    results: list[Path] = []
    for url in cfg.urls:
        try:
            path = run_one(url, cfg)
            results.append(path)
        except Exception as exc:
            logger.error("Failed to process %r: %s", url, exc)

    logger.info(
        "run_all: processed %d / %d URLs successfully",
        len(results),
        len(cfg.urls),
    )
    return results
