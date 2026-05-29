from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from onto_extractor.config import ExtractConfig
from onto_extractor.fetch import fetch_text
from onto_extractor.owl_builder import build_owl
from onto_extractor.preprocess import extract_nouns, split_sentences
from onto_extractor.relations import extract_cooccurrence, extract_isa
from onto_extractor.terms import select_terms

logger = logging.getLogger(__name__)


def _url_to_filename(url: str) -> str:
    """Derive a deterministic filename fragment from a URL."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"onto_{digest}.owl"


def run_one(url: str, cfg: ExtractConfig) -> Path:
    """Run the full extraction pipeline for a single URL.

    Pipeline stages:
    1. fetch     → raw text
    2. preprocess → sentences + noun lists
    3. terms     → concept candidates
    4. relations → is-a and co-occurrence pairs
    5. owl_builder → .owl file

    Args:
        url: Web page URL to process.
        cfg: Configuration controlling all pipeline parameters.

    Returns:
        Path to the written .owl file.
    """
    logger.info("=== Processing URL: %s ===", url)

    text = fetch_text(url)

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError(f"No sentences extracted from {url!r}")
    noun_lists = extract_nouns(sentences, tagger=cfg.tagger)

    concepts = select_terms(noun_lists, cfg)
    if not concepts:
        raise ValueError(
            f"No concept terms selected for {url!r} — "
            "consider lowering min_freq or min_term_len"
        )
    concept_set = set(concepts)

    isa_pairs = extract_isa(sentences, concept_set)
    cooc_pairs: list[tuple[str, str]] = []
    if cfg.use_cooccurrence:
        cooc_pairs = extract_cooccurrence(noun_lists, concept_set, cfg.cooc_min_count)

    out_path = Path(cfg.output_dir) / _url_to_filename(url)
    build_owl(concepts, isa_pairs, cooc_pairs, cfg, out_path)

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
