"""Unit tests for select_terms in terms.py.

All tests use synthetic noun lists — no network or KoNLPy required.
"""
from __future__ import annotations

import pytest

from onto_extractor.config import ExtractConfig
from onto_extractor.terms import select_terms


def _cfg(**overrides) -> ExtractConfig:
    defaults = dict(min_freq=2, min_term_len=2, top_terms=10)
    defaults.update(overrides)
    return ExtractConfig(**defaults)


class TestSelectTermsBasic:
    def test_returns_list_of_strings(self):
        noun_lists = [["사과", "과일", "맛"], ["사과", "과일", "색깔"]]
        result = select_terms(noun_lists, _cfg())
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_empty_input_returns_empty(self):
        assert select_terms([], _cfg()) == []

    def test_all_empty_sentences_returns_empty(self):
        assert select_terms([[], [], []], _cfg()) == []


class TestFrequencyFilter:
    def test_min_freq_filters_rare_terms(self):
        noun_lists = [
            ["사과", "과일"],
            ["사과", "과일"],
            ["딸기"],
        ]
        result = select_terms(noun_lists, _cfg(min_freq=2))
        assert "딸기" not in result
        assert "사과" in result
        assert "과일" in result

    def test_min_term_len_filters_short_terms(self):
        noun_lists = [
            ["사과", "나", "과일"],
            ["사과", "나", "과일"],
            ["사과", "나", "과일"],
        ]
        result = select_terms(noun_lists, _cfg(min_term_len=2, min_freq=2))
        assert "나" not in result
        assert "사과" in result


class TestTopTermsLimit:
    def test_top_terms_limit_respected(self):
        nouns = [f"단어{i:02d}" for i in range(20)]
        noun_lists = [nouns[:]] * 5
        result = select_terms(noun_lists, _cfg(top_terms=5, min_freq=2))
        assert len(result) <= 5

    def test_top_terms_larger_than_candidates(self):
        noun_lists = [["사과", "과일"]] * 3
        result = select_terms(noun_lists, _cfg(top_terms=100, min_freq=2))
        assert len(result) <= 2


class TestDeterminism:
    def test_same_input_same_output(self):
        noun_lists = [
            ["사과", "과일", "건강", "맛"],
            ["사과", "과일", "색깔", "건강"],
            ["사과", "과일", "건강", "영양"],
        ]
        cfg = _cfg(min_freq=2, top_terms=3)
        assert select_terms(noun_lists, cfg) == select_terms(noun_lists, cfg)

    def test_output_order_is_stable(self):
        noun_lists = [["가나", "나다", "다라"]] * 4 + [["가나", "나다"]] * 2
        cfg = _cfg(min_freq=2, top_terms=10, min_term_len=2)
        results = [select_terms(noun_lists, cfg) for _ in range(5)]
        assert all(r == results[0] for r in results)


class TestTfidfScoring:
    def test_tfidf_selects_terms_above_freq_threshold(self):
        # "전문어" appears only once → filtered by min_freq=2 → not selected
        # "공통어" appears 4 times → selected
        noun_lists = [
            ["공통어", "전문어"],
            ["공통어"],
            ["공통어"],
            ["공통어"],
        ]
        result = select_terms(noun_lists, _cfg(min_freq=2, top_terms=2))
        assert "공통어" in result
        assert "전문어" not in result  # freq=1 < min_freq=2

    def test_tfidf_ranks_by_mean_score(self):
        # "핵심어" appears in all 4 sentences (high document frequency → high mean TF-IDF)
        # "희귀어" appears in 2 sentences (lower mean contribution across 4 docs)
        # Mean TF-IDF rewards terms with more non-zero entries across docs
        noun_lists = [
            ["핵심어", "희귀어"],
            ["핵심어", "희귀어"],
            ["핵심어"],
            ["핵심어"],
        ]
        result = select_terms(noun_lists, _cfg(min_freq=2, top_terms=2))
        assert "핵심어" in result
        assert "희귀어" in result
        # 핵심어 appears in all 4 docs → higher mean TF-IDF
        assert result.index("핵심어") < result.index("희귀어")
