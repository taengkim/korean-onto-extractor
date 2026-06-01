"""Unit tests for Lexicon loading and query interface.

Uses the package-bundled lexicons/ directory — no network or KoNLPy required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from onto_extractor.lexicon import Lexicon

BUNDLED_LEXICONS = Path(__file__).parent.parent / "onto_extractor" / "lexicons"


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return Lexicon.load(BUNDLED_LEXICONS)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

class TestLexiconLoad:
    def test_loads_without_error(self, lexicon):
        assert lexicon is not None

    def test_entities_populated(self, lexicon):
        assert len(lexicon.entities) > 0
        assert len(lexicon._entity_set) > 0

    def test_stopwords_populated(self, lexicon):
        assert len(lexicon.stopwords) > 0

    def test_unit_map_populated(self, lexicon):
        assert len(lexicon.unit_map) > 0

    def test_relation_patterns_populated(self, lexicon):
        assert len(lexicon.relation_patterns) > 0

    def test_triggers_populated(self, lexicon):
        assert len(lexicon.triggers) > 0

    def test_empty_returns_noop(self):
        empty = Lexicon.empty()
        assert not empty.is_stopword("것")
        assert not empty.is_noise("123")
        assert empty.normalize("kg") == "kg"
        assert not empty.is_entity("인공지능")
        assert empty.relation_patterns == []


# ------------------------------------------------------------------
# 개체 사전 (Entity Dictionary)
# ------------------------------------------------------------------

class TestEntityDictionary:
    def test_known_entity_recognized(self, lexicon):
        assert lexicon.is_entity("인공지능")
        assert lexicon.is_entity("물리학")

    def test_unknown_term_not_entity(self, lexicon):
        assert not lexicon.is_entity("전혀없는단어XYZ")

    def test_entity_category_returned(self, lexicon):
        category = lexicon.entity_category("인공지능")
        assert category == "기술"

    def test_entity_category_none_for_unknown(self, lexicon):
        assert lexicon.entity_category("없는단어") is None


# ------------------------------------------------------------------
# 트리거 사전 (Trigger Dictionary)
# ------------------------------------------------------------------

class TestTriggerDictionary:
    def test_trigger_sentence_detected(self, lexicon):
        sent = "고양이는 동물의 일종이다."
        assert lexicon.is_trigger_sentence(sent)

    def test_non_trigger_sentence(self, lexicon):
        sent = "오늘 날씨가 맑다."
        assert not lexicon.is_trigger_sentence(sent)

    def test_trigger_types_returned(self, lexicon):
        sent = "사과는 과일의 일종이다."
        types = lexicon.trigger_types(sent)
        assert "is_a" in types


# ------------------------------------------------------------------
# 단위 정규화 사전 (Unit Normalization)
# ------------------------------------------------------------------

class TestUnitNormalization:
    def test_unit_variant_normalized(self, lexicon):
        assert lexicon.normalize("kg") == "킬로그램"
        assert lexicon.normalize("KG") == "킬로그램"  # case-insensitive

    def test_unknown_unit_unchanged(self, lexicon):
        assert lexicon.normalize("사과") == "사과"

    def test_percent_normalized(self, lexicon):
        assert lexicon.normalize("%") == "퍼센트"


# ------------------------------------------------------------------
# 불용어 사전 (Stopword Dictionary)
# ------------------------------------------------------------------

class TestStopwords:
    def test_known_stopword(self, lexicon):
        assert lexicon.is_stopword("것")
        assert lexicon.is_stopword("수")
        assert lexicon.is_stopword("등")

    def test_domain_term_not_stopword(self, lexicon):
        assert not lexicon.is_stopword("인공지능")
        assert not lexicon.is_stopword("물리학")


# ------------------------------------------------------------------
# 노이즈 패턴 (Noise Patterns)
# ------------------------------------------------------------------

class TestNoisePatterns:
    def test_pure_number_is_noise(self, lexicon):
        assert lexicon.is_noise("123")
        assert lexicon.is_noise("42")

    def test_year_is_noise(self, lexicon):
        assert lexicon.is_noise("2024년")

    def test_month_is_noise(self, lexicon):
        assert lexicon.is_noise("3월")

    def test_normal_korean_not_noise(self, lexicon):
        assert not lexicon.is_noise("인공지능")
        assert not lexicon.is_noise("물리학")

    def test_single_latin_char_is_noise(self, lexicon):
        assert lexicon.is_noise("A")
        assert lexicon.is_noise("b")


# ------------------------------------------------------------------
# 관계 트리거 사전 통합: extract_typed_relations
# ------------------------------------------------------------------

class TestRelationTriggers:
    def test_hasPart_pattern_fires(self, lexicon):
        from onto_extractor.relations import extract_typed_relations

        sent = "컴퓨터는 반도체로 구성된다."
        concepts = {"컴퓨터", "반도체"}
        triples = extract_typed_relations([sent], concepts, lexicon)
        assert any(
            subj == "컴퓨터" and rel == "hasPart" and obj == "반도체"
            for subj, rel, obj in triples
        ), f"Expected hasPart triple not found. Got: {triples}"

    def test_madeOf_pattern_fires(self, lexicon):
        from onto_extractor.relations import extract_typed_relations

        sent = "그릇은 도자기로 만들어진다."
        concepts = {"그릇", "도자기"}
        triples = extract_typed_relations([sent], concepts, lexicon)
        assert any(rel == "madeOf" for _, rel, _ in triples), \
            f"Expected madeOf triple not found. Got: {triples}"

    def test_no_match_outside_concepts(self, lexicon):
        from onto_extractor.relations import extract_typed_relations

        sent = "컴퓨터는 반도체로 구성된다."
        # 반도체 not in concepts → no triple
        triples = extract_typed_relations([sent], {"컴퓨터"}, lexicon)
        assert len(triples) == 0

    def test_empty_lexicon_returns_empty(self):
        from onto_extractor.relations import extract_typed_relations

        empty = Lexicon.empty()
        triples = extract_typed_relations(["테스트 문장이다."], {"테스트"}, empty)
        assert triples == []


# ------------------------------------------------------------------
# normalize_nouns integration
# ------------------------------------------------------------------

class TestNormalizeNouns:
    def test_noise_tokens_removed(self, lexicon):
        from onto_extractor.preprocess import normalize_nouns

        noun_lists = [["인공지능", "2024년", "연구"]]
        result = normalize_nouns(noun_lists, lexicon)
        assert result == [["인공지능", "연구"]]

    def test_unit_variants_normalized(self, lexicon):
        from onto_extractor.preprocess import normalize_nouns

        noun_lists = [["무게", "kg", "측정"]]
        result = normalize_nouns(noun_lists, lexicon)
        assert "킬로그램" in result[0]
        assert "kg" not in result[0]

    def test_empty_input(self, lexicon):
        from onto_extractor.preprocess import normalize_nouns

        assert normalize_nouns([], lexicon) == []
        assert normalize_nouns([[]], lexicon) == [[]]


# ------------------------------------------------------------------
# select_terms stopword filtering
# ------------------------------------------------------------------

class TestSelectTermsWithLexicon:
    def test_stopwords_excluded(self, lexicon):
        from onto_extractor.config import ExtractConfig
        from onto_extractor.terms import select_terms

        cfg = ExtractConfig(min_freq=1, min_term_len=1, top_terms=10)
        # "것", "수" are stopwords
        noun_lists = [["것", "수", "인공지능"], ["것", "수", "인공지능"]]
        result = select_terms(noun_lists, cfg, lexicon=lexicon)
        assert "것" not in result
        assert "수" not in result
        assert "인공지능" in result

    def test_entity_injected_at_min_freq(self, lexicon):
        from onto_extractor.config import ExtractConfig
        from onto_extractor.terms import select_terms

        cfg = ExtractConfig(min_freq=2, min_term_len=2, top_terms=50)
        # "인공지능" appears twice → should be in result via entity injection
        noun_lists = [["인공지능", "연구"], ["인공지능", "결과"]]
        result = select_terms(noun_lists, cfg, lexicon=lexicon)
        assert "인공지능" in result
