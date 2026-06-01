from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import yaml

logger = logging.getLogger(__name__)

# Default lexicon directory bundled with the package
_PACKAGE_LEXICONS = Path(__file__).parent / "lexicons"


class RelationPattern(NamedTuple):
    relation_type: str
    pattern: re.Pattern
    swapped: bool


class Lexicon:
    """Container for all lexicon resources used by the extraction pipeline.

    Acts as a null object when constructed via ``Lexicon.empty()`` — all
    queries return safe defaults so callers need no None checks.
    """

    def __init__(
        self,
        entities: dict[str, list[str]] | None = None,
        triggers: dict[str, set[str]] | None = None,
        unit_map: dict[str, str] | None = None,
        relation_patterns: list[RelationPattern] | None = None,
        stopwords: set[str] | None = None,
        noise_patterns: list[re.Pattern] | None = None,
    ) -> None:
        # 개체 사전
        self.entities: dict[str, list[str]] = entities or {}
        # 트리거 사전
        self.triggers: dict[str, set[str]] = triggers or {}
        # 단위 정규화 사전 (variant → canonical)
        self.unit_map: dict[str, str] = unit_map or {}
        # 관계 트리거 사전
        self.relation_patterns: list[RelationPattern] = relation_patterns or []
        # 불용어 사전
        self.stopwords: set[str] = stopwords or set()
        # 노이즈 패턴
        self.noise_patterns: list[re.Pattern] = noise_patterns or []

        # Derived O(1) lookup structures
        self._entity_set: set[str] = set()
        self._entity_category_map: dict[str, str] = {}
        for category, entity_list in self.entities.items():
            for entity in entity_list:
                self._entity_set.add(entity)
                self._entity_category_map[entity] = category

        self._all_triggers: set[str] = set()
        for words in self.triggers.values():
            self._all_triggers.update(words)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def is_stopword(self, term: str) -> bool:
        return term in self.stopwords

    def is_noise(self, term: str) -> bool:
        return any(p.fullmatch(term) for p in self.noise_patterns)

    def normalize(self, term: str) -> str:
        """Return the canonical form of a unit variant, or the term itself."""
        return self.unit_map.get(term.lower(), term)

    def is_entity(self, term: str) -> bool:
        return term in self._entity_set

    def entity_category(self, term: str) -> str | None:
        return self._entity_category_map.get(term)

    def is_trigger_sentence(self, sentence: str) -> bool:
        """Return True if the sentence contains any trigger word."""
        return any(trigger in sentence for trigger in self._all_triggers)

    def trigger_types(self, sentence: str) -> list[str]:
        """Return the trigger type names found in a sentence."""
        return [
            trigger_type
            for trigger_type, words in self.triggers.items()
            if any(w in sentence for w in words)
        ]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, lexicon_dir: str | Path | None = None) -> "Lexicon":
        """Load all lexicon YAML files from *lexicon_dir*.

        Falls back to the package-bundled ``lexicons/`` directory when
        *lexicon_dir* is None.  Missing individual files are skipped with
        a warning so partial lexicons work correctly.
        """
        lex_path = Path(lexicon_dir) if lexicon_dir else _PACKAGE_LEXICONS
        if not lex_path.exists():
            logger.warning(
                "Lexicon directory not found: %s — using empty lexicon", lex_path
            )
            return cls.empty()

        def _load(filename: str) -> dict:
            p = lex_path / filename
            if not p.exists():
                logger.warning("Lexicon file missing: %s", p)
                return {}
            with p.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

        # --- 개체 사전 ---
        entities: dict[str, list[str]] = {}
        for category, items in _load("entities.yaml").items():
            entities[category] = [str(e) for e in (items or [])]

        # --- 트리거 사전 ---
        triggers: dict[str, set[str]] = {}
        for ttype, words in _load("triggers.yaml").items():
            triggers[ttype] = set(str(w) for w in (words or []))

        # --- 단위 정규화 사전 ---
        unit_map: dict[str, str] = {}
        for canonical, variants in _load("units.yaml").items():
            canonical_str = str(canonical)
            for v in (variants or []):
                unit_map[str(v).lower()] = canonical_str

        # --- 관계 트리거 사전 ---
        relation_patterns: list[RelationPattern] = []
        for rel_type, entries in _load("relation_triggers.yaml").items():
            for entry in (entries or []):
                if isinstance(entry, dict):
                    raw_pat = str(entry.get("pattern", "")).strip()
                    swapped = bool(entry.get("swapped", False))
                else:
                    raw_pat = str(entry).strip()
                    swapped = False
                if not raw_pat:
                    continue
                try:
                    compiled = re.compile(raw_pat, re.UNICODE)
                    relation_patterns.append(
                        RelationPattern(str(rel_type), compiled, swapped)
                    )
                except re.error as exc:
                    logger.warning(
                        "Invalid relation pattern in %s: %r — %s", rel_type, raw_pat, exc
                    )

        # --- 불용어 및 노이즈 사전 ---
        sw_data = _load("stopwords.yaml")
        stopwords: set[str] = set(str(w) for w in sw_data.get("stopwords", []))
        noise_patterns: list[re.Pattern] = []
        for raw in sw_data.get("noise_patterns", []):
            try:
                noise_patterns.append(re.compile(str(raw), re.UNICODE))
            except re.error as exc:
                logger.warning("Invalid noise pattern %r: %s", raw, exc)

        lexicon = cls(
            entities=entities,
            triggers=triggers,
            unit_map=unit_map,
            relation_patterns=relation_patterns,
            stopwords=stopwords,
            noise_patterns=noise_patterns,
        )
        logger.info(
            "Lexicon loaded from %s: %d entities, %d trigger words, "
            "%d unit mappings, %d relation patterns, %d stopwords, %d noise patterns",
            lex_path,
            len(lexicon._entity_set),
            len(lexicon._all_triggers),
            len(unit_map),
            len(relation_patterns),
            len(stopwords),
            len(noise_patterns),
        )
        return lexicon

    @classmethod
    def empty(cls) -> "Lexicon":
        """Return a no-op Lexicon (all lookups return safe defaults)."""
        return cls()
