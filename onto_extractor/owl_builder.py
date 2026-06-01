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
    typed_pairs: list[tuple[str, str, str]] | None = None,
) -> Path:
    """Build an OWL ontology and write it to out_path.

    Structure:
    - Each concept → OWL Class (IRI = base_iri + url_encoded_term)
    - is-a pair (hypo, hyper) → rdfs:subClassOf
    - typed triple (subj, rel_type, obj) → ObjectProperty SomeValuesFrom restriction
    - co-occurrence pair (a, b) → AnnotationProperty relatedTo

    Args:
        concepts: Ordered list of concept strings.
        isa_pairs: List of (hyponym, hypernym) tuples.
        cooc_pairs: List of (concept_a, concept_b) co-occurrence pairs.
        cfg: Configuration (supplies base_iri).
        out_path: Output file path (.owl).
        typed_pairs: Optional list of (subject, relation_type, object) triples
            from relation trigger patterns. Each unique relation_type becomes
            an OWL ObjectProperty.

    Returns:
        out_path for chaining/logging.
    """
    try:
        import owlready2
    except ImportError as exc:
        raise ImportError(
            "owlready2 is required for OWL output. Install it with: pip install owlready2"
        ) from exc

    typed_pairs = typed_pairs or []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    onto_iri = cfg.base_iri.rstrip("#/") + "/"
    onto = owlready2.get_ontology(onto_iri)

    with onto:
        # Annotation property for co-occurrence (semantically weak, no range/domain)
        class relatedTo(owlready2.AnnotationProperty):  # noqa: N801
            pass

        # Create OWL ObjectProperties for typed relation types
        obj_props: dict[str, type] = {}
        for _, rel_type, _ in typed_pairs:
            if rel_type not in obj_props:
                safe_rel = _safe_iri_fragment(rel_type)
                prop_cls = owlready2.types.new_class(
                    safe_rel, (owlready2.ObjectProperty,)
                )
                obj_props[rel_type] = prop_cls
                logger.debug("OWL ObjectProperty created: %s", rel_type)

        # Create OWL Class for each concept
        class_map: dict[str, owlready2.ThingClass] = {}
        for term in sorted(concepts):
            safe_name = _safe_iri_fragment(term)
            cls = owlready2.types.new_class(safe_name, (owlready2.Thing,))
            cls.label = [term]
            class_map[term] = cls

        # is-a → rdfs:subClassOf
        for hypo, hyper in isa_pairs:
            if hypo in class_map and hyper in class_map:
                class_map[hypo].is_a.append(class_map[hyper])
                logger.debug("OWL subClassOf: %s → %s", hypo, hyper)

        # typed relations → ObjectProperty existential restriction
        for subj, rel_type, obj in typed_pairs:
            if subj in class_map and obj in class_map and rel_type in obj_props:
                prop = obj_props[rel_type]
                restriction = prop.some(class_map[obj])
                class_map[subj].is_a.append(restriction)
                logger.debug("OWL %s.some(%s) on %s", rel_type, obj, subj)

        # co-occurrence → relatedTo annotation
        for a, b in cooc_pairs:
            if a in class_map and b in class_map:
                class_map[a].relatedTo.append(class_map[b])
                logger.debug("OWL relatedTo: %s ↔ %s", a, b)

    onto.save(file=str(out_path), format="rdfxml")
    logger.info(
        "Saved ontology to %s (%d classes, %d object properties)",
        out_path, len(class_map), len(obj_props),
    )
    return out_path
