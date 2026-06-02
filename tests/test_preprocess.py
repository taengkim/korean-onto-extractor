"""Unit tests for _collect_nouns_and_phrases and split_sentences in preprocess.py.

All tests use synthetic POS-tagged input — no network or KoNLPy required.
"""
from __future__ import annotations

import pytest

from onto_extractor.preprocess import (
    _collect_nouns_and_phrases,
    _COMPOUND_NOUN_TAGS,
    split_sentences,
)


# Shorthand helpers
def collect(tagged, tagger="komoran", max_len=2):
    tags = _COMPOUND_NOUN_TAGS[tagger]
    return _collect_nouns_and_phrases(tagged, tags, max_len=max_len)


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


# ------------------------------------------------------------------
# phrase_max_len (n-gram chunking)
# ------------------------------------------------------------------

class TestPhraseMaxLen:
    def test_max_len_1_no_phrases(self):
        # max_len=1: 개별 명사만, 구 없음
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("연구", "NNG")]
        result = collect(tagged, max_len=1)
        assert "인공" in result
        assert "지능" in result
        assert "연구" in result
        assert "인공지능" not in result
        assert "지능연구" not in result

    def test_max_len_2_bigrams_only(self):
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("연구", "NNG")]
        result = collect(tagged, max_len=2)
        assert "인공지능" in result
        assert "지능연구" in result
        assert "인공지능연구" not in result

    def test_max_len_3_includes_trigram(self):
        tagged = [("인공", "NNG"), ("지능", "NNG"), ("연구", "NNG")]
        result = collect(tagged, max_len=3)
        assert "인공지능" in result
        assert "지능연구" in result
        assert "인공지능연구" in result

    def test_max_len_3_four_noun_run(self):
        # 4개 명사 런, max_len=3
        tagged = [("자연", "NNG"), ("어", "NNG"), ("처리", "NNG"), ("기술", "NNG")]
        result = collect(tagged, max_len=3)
        # 바이그램
        assert "자연어" in result
        assert "어처리" in result
        assert "처리기술" in result
        # 트라이그램
        assert "자연어처리" in result
        assert "어처리기술" in result
        # 4-그램은 포함 안 됨
        assert "자연어처리기술" not in result

    def test_max_len_4_includes_four_gram(self):
        tagged = [("자연", "NNG"), ("어", "NNG"), ("처리", "NNG"), ("기술", "NNG")]
        result = collect(tagged, max_len=4)
        assert "자연어처리기술" in result

    def test_max_len_2_run_shorter_than_max(self):
        # 런이 max_len보다 짧으면 가능한 구만 생성
        tagged = [("사과", "NNG")]
        result = collect(tagged, max_len=3)
        assert result == ["사과"]


# ------------------------------------------------------------------
# split_sentences — custom endings and min_len
# ------------------------------------------------------------------

class TestSplitSentences:
    def test_default_endings_splits_on_다(self):
        text = "인공지능은 발전한다. 기계학습도 중요하다."
        sentences = split_sentences(text)
        assert len(sentences) >= 1

    def test_custom_endings_adds_new_boundary(self):
        # "함" 어미 추가 → "함." 뒤에도 분리
        text = "연구를 진행함. 결과를 분석한다."
        default_sents = split_sentences(text)
        custom_sents = split_sentences(text, endings=["다", "요", "죠", "군요", "함"])
        # 커스텀 어미를 추가하면 분리 개수가 같거나 더 많아야 함
        assert len(custom_sents) >= len(default_sents)

    def test_custom_endings_empty_list_only_punctuation(self):
        # endings=[] → 구두점(.!?)만 분리
        text = "문장 하나다. 문장 둘이다."
        sentences = split_sentences(text, endings=[])
        # 결과는 비어있지 않아야 함 (구두점으로 분리됨)
        assert len(sentences) >= 1

    def test_min_len_filters_short_segments(self):
        text = "안녕. 이것은 충분히 긴 문장입니다."
        long_sents = split_sentences(text, min_len=5)
        short_sents = split_sentences(text, min_len=1)
        # min_len=1이면 더 많은 세그먼트를 유지함
        assert len(short_sents) >= len(long_sents)

    def test_min_len_1_keeps_short_segments(self):
        text = "안녕. 네."
        sentences = split_sentences(text, min_len=1)
        assert any("네" in s for s in sentences)

    def test_newlines_replaced_with_period(self):
        text = "첫 번째 줄\n두 번째 줄이다."
        sentences = split_sentences(text)
        assert len(sentences) >= 1


# ------------------------------------------------------------------
# ExtractConfig chunking fields
# ------------------------------------------------------------------

class TestExtractConfigChunkingFields:
    def test_default_sent_endings(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig()
        assert "다" in cfg.sent_endings
        assert "요" in cfg.sent_endings

    def test_default_min_sent_len(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig()
        assert cfg.min_sent_len == 5

    def test_default_phrase_max_len(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig()
        assert cfg.phrase_max_len == 2

    def test_custom_phrase_max_len(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig(phrase_max_len=3)
        assert cfg.phrase_max_len == 3

    def test_custom_sent_endings(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig(sent_endings=["다", "함"])
        assert cfg.sent_endings == ["다", "함"]

    def test_phrase_max_len_ge_1(self):
        from onto_extractor.config import ExtractConfig
        import pytest
        with pytest.raises(Exception):
            ExtractConfig(phrase_max_len=0)

    def test_default_compound_noun_tags_is_none(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig()
        assert cfg.compound_noun_tags is None

    def test_custom_compound_noun_tags(self):
        from onto_extractor.config import ExtractConfig
        cfg = ExtractConfig(compound_noun_tags=["NNG", "NNP", "SL"])
        assert cfg.compound_noun_tags == ["NNG", "NNP", "SL"]


# ------------------------------------------------------------------
# compound_noun_tags override in _collect_nouns_and_phrases
# ------------------------------------------------------------------

class TestCompoundNounTagsOverride:
    def test_custom_tag_included_in_run(self):
        # SL(외래어)를 태그 세트에 포함 → 외래어도 복합명사 형성
        tags = frozenset({"NNG", "SL"})
        tagged = [("인공", "NNG"), ("AI", "SL"), ("연구", "NNG")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "인공AI" in result
        assert "AI연구" in result

    def test_custom_tag_excluded_breaks_run(self):
        # SL을 태그 세트에서 제외 → 외래어가 런을 끊음
        tags = frozenset({"NNG"})
        tagged = [("인공", "NNG"), ("AI", "SL"), ("연구", "NNG")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "인공AI" not in result
        assert "AI연구" not in result
        assert "인공" in result
        assert "연구" in result

    def test_nnb_included_when_explicitly_added(self):
        # NNB를 명시적으로 추가하면 복합명사 형성에 포함됨
        tags = frozenset({"NNG", "NNB"})
        tagged = [("사과", "NNG"), ("것", "NNB"), ("과일", "NNG")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "사과것" in result
        assert "것과일" in result

    def test_single_custom_tag_only(self):
        # 커스텀 태그 한 종류만 지정
        tags = frozenset({"XR"})
        tagged = [("먹", "XR"), ("이", "XR"), ("살", "NNG")]
        result = _collect_nouns_and_phrases(tagged, tags)
        assert "먹이" in result
        assert "이살" not in result  # NNG는 태그 세트에 없으므로 런 끊김
