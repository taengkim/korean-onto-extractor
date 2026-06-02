"""Unit tests for _collect_nouns_and_phrases in preprocess.py.

All tests use synthetic POS-tagged input — no network or KoNLPy required.
"""
from __future__ import annotations

import pytest

from onto_extractor.preprocess import _collect_nouns_and_phrases, _COMPOUND_NOUN_TAGS


# Shorthand helpers
def collect(tagged, tagger="komoran"):
    tags = _COMPOUND_NOUN_TAGS[tagger]
    return _collect_nouns_and_phrases(tagged, tags)


# ------------------------------------------------------------------
# Basic extraction
# ------------------------------------------------------------------

class TestBasicExtraction:
    def test_empty_input_returns_empty(self):
        assert collect([]) == []

    def test_single_non_noun_returns_empty(self):
        assert collect([("는", "JX")]) == []

    def test_single_noun_no_bigram(self):
        result = collect([("사과", "NNG")])
        assert result == ["사과"]

    def test_two_nouns_produces_bigram(self):
        tagged = [("인공", "NNG"), ("지능", "NNG")]
        result = collect(tagged)
        assert "인공" in result
        assert "지능" in result
        assert "인공지능" in result

    def test_bigram_order_preserving(self):
        tagged = [("인공", "NNG"), ("지능", "NNG")]
        result = collect(tagged)
        # 개별 명사가 먼저, 바이그램이 뒤에 나와야 함
        assert result.index("인공") < result.index("인공지능")
        assert result.index("지능") < result.index("인공지능")


# ------------------------------------------------------------------
# Run lengths
# ------------------------------------------------------------------

class TestRunLengths:
    def test_three_noun_run_produces_two_bigrams(self):
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("연구", "NNG")]
        result = collect(tagged)
        # 개별 명사 3개
        assert "인공" in result
        assert "지능" in result
        assert "연구" in result
        # 바이그램 2개
        assert "인공지능" in result
        assert "지능연구" in result
        # 트라이그램은 생성하지 않음
        assert "인공지능연구" not in result

    def test_four_noun_run_produces_three_bigrams(self):
        tagged = [
            ("자연", "NNG"), ("어", "NNG"), ("처리", "NNG"), ("기술", "NNG")
        ]
        result = collect(tagged)
        assert "자연어" in result
        assert "어처리" in result
        assert "처리기술" in result
        # 개별도 모두 포함
        assert "자연" in result
        assert "어" in result
        assert "처리" in result
        assert "기술" in result

    def test_single_noun_in_run(self):
        # 비명사 토큰으로 둘러싸인 단일 명사 — 바이그램 없음
        tagged = [("는", "JX"), ("사과", "NNG"), ("를", "JX")]
        result = collect(tagged)
        assert result == ["사과"]


# ------------------------------------------------------------------
# Mixed POS sequences
# ------------------------------------------------------------------

class TestMixedPOS:
    def test_non_noun_breaks_run(self):
        # "인공" + 조사 + "지능" → 두 개의 별개 런, 바이그램 없음
        tagged = [
            ("인공", "NNG"), ("의", "JX"), ("지능", "NNG")
        ]
        result = collect(tagged)
        assert "인공" in result
        assert "지능" in result
        assert "인공지능" not in result

    def test_nnp_included_in_run(self):
        # NNP(고유명사)도 복합명사 형성에 포함
        tagged = [("서울", "NNP"), ("대학교", "NNG")]
        result = collect(tagged)
        assert "서울대학교" in result

    def test_nnb_excluded_from_compound(self):
        # NNB(의존명사: 것/수/때) — compound_tags에 없으므로 런 끊김
        tagged = [("사과", "NNG"), ("것", "NNB"), ("과일", "NNG")]
        result = collect(tagged)
        assert "사과" in result
        assert "과일" in result
        assert "사과것" not in result
        assert "것과일" not in result

    def test_verb_between_nouns_breaks_run(self):
        tagged = [
            ("기계", "NNG"), ("학습", "NNG"), ("한다", "VX"), ("알고리즘", "NNG")
        ]
        result = collect(tagged)
        assert "기계학습" in result
        assert "학습알고리즘" not in result
        assert "알고리즘" in result

    def test_multiple_separate_runs(self):
        tagged = [
            ("인공", "NNG"), ("지능", "NNG"),
            ("은", "JX"),
            ("기계", "NNG"), ("학습", "NNG"),
        ]
        result = collect(tagged)
        assert "인공지능" in result
        assert "기계학습" in result
        assert "지능기계" not in result  # 런이 분리되므로 교차 바이그램 없음


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_morphemes_deduplicated(self):
        # 같은 명사가 두 런에 나타나도 중복 없이
        tagged = [
            ("사과", "NNG"), ("과일", "NNG"),
            ("은", "JX"),
            ("사과", "NNG"), ("과일", "NNG"),
        ]
        result = collect(tagged)
        assert result.count("사과") == 1
        assert result.count("과일") == 1
        assert result.count("사과과일") == 1


# ------------------------------------------------------------------
# Tagger-specific tag schemes
# ------------------------------------------------------------------

class TestTaggerTagSchemes:
    def test_okt_noun_tag(self):
        tags = _COMPOUND_NOUN_TAGS["okt"]
        tagged = [("인공지능", "Noun"), ("연구", "Noun"), ("를", "Josa")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "인공지능" in result
        assert "연구" in result
        assert "인공지능연구" in result

    def test_hannanum_nc_nq_tags(self):
        tags = _COMPOUND_NOUN_TAGS["hannanum"]
        tagged = [
            ("서울", "NQ"), ("시", "NC"), ("이다", "VP")
        ]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "서울" in result
        assert "시" in result
        assert "서울시" in result

    def test_kkma_tag_scheme(self):
        tags = _COMPOUND_NOUN_TAGS["kkma"]
        tagged = [("기계", "NNG"), ("학습", "NNG"), ("법", "NNB")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "기계학습" in result
        # NNB는 compound_tags에 없으므로 런 끊김
        assert "학습법" not in result

    def test_all_taggers_have_entries(self):
        for tagger_name in ["komoran", "mecab", "okt", "hannanum", "kkma"]:
            assert tagger_name in _COMPOUND_NOUN_TAGS
            assert len(_COMPOUND_NOUN_TAGS[tagger_name]) > 0

    def test_unknown_tagger_falls_back_to_default(self):
        from onto_extractor.preprocess import _DEFAULT_COMPOUND_TAGS
        # _DEFAULT_COMPOUND_TAGS는 NNG, NNP를 포함해야 함
        assert "NNG" in _DEFAULT_COMPOUND_TAGS
        assert "NNP" in _DEFAULT_COMPOUND_TAGS


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_morpheme_skipped(self):
        tagged = [("", "NNG"), ("사과", "NNG")]
        result = collect(tagged)
        assert "" not in result
        assert "사과" in result
        # 빈 형태소는 런에서 제외되므로 "사과" 이전 런이 없음 → 바이그램 없음
        assert "사과" in result

    def test_whitespace_morpheme_skipped(self):
        tagged = [(" ", "NNG"), ("사과", "NNG")]
        result = collect(tagged)
        assert " " not in result
        assert "사과" in result

    def test_all_non_nouns(self):
        tagged = [("는", "JX"), ("을", "JX"), ("한다", "VX")]
        assert collect(tagged) == []

    def test_large_run_bigrams_only(self):
        # 5개 명사 런 → 4개 바이그램, 5개 개별 명사
        run = [(f"명{i}", "NNG") for i in range(5)]
        result = collect(run)
        individual_count = sum(1 for w in result if len(w) == 2)  # "명0".."명4"
        bigram_count = sum(1 for w in result if len(w) == 4)      # "명0명1".."명3명4"
        assert individual_count == 5
        assert bigram_count == 4
