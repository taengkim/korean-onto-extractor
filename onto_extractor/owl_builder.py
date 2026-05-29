from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path

from onto_extractor.config import ExtractConfig

logger = logging.getLogger(__name__)


def _safe_iri_fragment(term: str) -> str:
    """Encode a Korean/mixed term for safe use as an IRI fragment."""
    safe_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )
    return urllib.parse.quote(term, safe=safe_chars)


def build_owl(
    concepts: list[str],
    isa_pairs: list[tuple[str, str]],
    cooc_pairs: list[tuple[str, str]],
    cfg: ExtractConfig,
    out_path: Path,
) -> Path:
    """Build an OWL ontology and write it to out_path.

    Structure:
    - Each concept → OWL Class (IRI = base_iri + url_encoded_term)
    - is-a pair (hypo, hyper) → rdfs:subClassOf assertion
    - co-occurrence pair (a, b) → annotation with relatedTo property

    Args:
        concepts: Ordered list of concept strings.
        isa_pairs: List of (hyponym, hypernym) tuples.
        cooc_pairs: List of (concept_a, concept_b) co-occurrence pairs.
        cfg: Configuration (supplies base_iri).
        out_path: Output file path (.owl).

    Returns:
        out_path for chaining/logging.
    """
    try:
        import owlready2
    except ImportError as exc:
        raise ImportError(
            "owlready2 is required for OWL output. Install it with: pip install owlready2"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    onto_iri = cfg.base_iri.rstrip("#/") + "/"
    onto = owlready2.get_ontology(onto_iri)

    with onto:
        class relatedTo(owlready2.AnnotationProperty):  # noqa: N801
            pass

        class_map: dict[str, owlready2.ThingClass] = {}
        for term in sorted(concepts):
            safe_name = _safe_iri_fragment(term)
            cls = owlready2.types.new_class(safe_name, (owlready2.Thing,))
            cls.label = [term]
            class_map[term] = cls

        for hypo, hyper in isa_pairs:
            if hypo in class_map and hyper in class_map:
                class_map[hypo].is_a.append(class_map[hyper])
                logger.debug("OWL subClassOf: %s → %s", hypo, hyper)

        for a, b in cooc_pairs:
            if a in class_map and b in class_map:
                class_map[a].relatedTo.append(class_map[b])
                logger.debug("OWL relatedTo: %s ↔ %s", a, b)

    onto.save(file=str(out_path), format="rdfxml")
    logger.info("Saved ontology to %s (%d classes)", out_path, len(class_map))
    return out_path
