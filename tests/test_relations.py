"""Unit tests for Korean Hearst pattern extraction in relations.py.

All tests are pure regex — no network calls, no KoNLPy required.
"""
from __future__ import annotations

import pytest

from onto_extractor.relations import extract_cooccurrence, extract_isa


def isa(sentences, concepts):
    return extract_isa(sentences, set(concepts))


class TestPatternIljongida:
    def test_basic(self):
        sent = "고양이는 동물의 일종이다."
        assert ("고양이", "동물") in isa([sent], ["고양이", "동물"])

    def test_multiple_occurrences(self):
        sents = ["고양이는 동물의 일종이다.", "장미는 꽃의 일종이다."]
        pairs = isa(sents, ["고양이", "동물", "장미", "꽃"])
        assert ("고양이", "동물") in pairs
        assert ("장미", "꽃") in pairs

    def test_missing_concept_excluded(self):
        sent = "고양이는 동물의 일종이다."
        assert len(isa([sent], ["고양이"])) == 0


class TestPatternGateun:
    def test_basic_with_particle(self):
        sent = "철수와 같은 학생이 많다."
        assert ("철수", "학생") in isa([sent], ["철수", "학생"])

    def test_no_particle(self):
        sent = "장미 같은 꽃을 좋아한다."
        assert ("장미", "꽃") in isa([sent], ["장미", "꽃"])


class TestPatternPohamDoenda:
    def test_basic(self):
        sent = "과일에는 사과가 포함된다."
        assert ("사과", "과일") in isa([sent], ["사과", "과일"])

    def test_without_neun(self):
        sent = "동물에 고양이가 포함된다."
        assert ("고양이", "동물") in isa([sent], ["고양이", "동물"])


class TestPatternIda:
    def test_basic(self):
        sent = "장미는 꽃이다."
        assert ("장미", "꽃") in isa([sent], ["장미", "꽃"])

    def test_imnida_form(self):
        sent = "독수리는 조류입니다."
        assert ("독수리", "조류") in isa([sent], ["독수리", "조류"])


class TestPatternDeung:
    def test_basic_list(self):
        sent = "사과, 배 등 과일이 있다."
        pairs = isa([sent], ["사과", "배", "과일"])
        assert ("배", "과일") in pairs

    def test_single_item(self):
        sent = "사과 등의 과일을 먹자."
        assert ("사과", "과일") in isa([sent], ["사과", "과일"])


class TestPatternBullineun:
    def test_basic(self):
        sent = "인공지능이라고 불리는 기술이 발전했다."
        assert ("인공지능", "기술") in isa([sent], ["인공지능", "기술"])


class TestPatternHanaIn:
    def test_basic(self):
        sent = "과학의 하나인 물리학을 공부한다."
        assert ("물리학", "과학") in isa([sent], ["물리학", "과학"])


class TestDeduplication:
    def test_duplicate_sentences_deduplicated(self):
        sents = ["고양이는 동물의 일종이다.", "고양이는 동물의 일종이다."]
        pairs = isa(sents, ["고양이", "동물"])
        assert pairs.count(("고양이", "동물")) == 1

    def test_reflexive_excluded(self):
        sent = "고양이는 고양이의 일종이다."
        assert len(isa([sent], ["고양이"])) == 0


class TestCooccurrence:
    def test_basic(self):
        noun_lists = [
            ["사과", "과일", "건강"],
            ["사과", "과일", "색깔"],
            ["사과", "과일", "맛"],
        ]
        pairs = extract_cooccurrence(noun_lists, {"사과", "과일", "건강"}, min_count=2)
        # canonical order: "과일" < "사과" in Unicode, so pair is ("과일", "사과")
        assert ("과일", "사과") in pairs

    def test_below_threshold_excluded(self):
        noun_lists = [["사과", "배"], ["사과", "포도"]]
        pairs = extract_cooccurrence(noun_lists, {"사과", "배", "포도"}, min_count=3)
        assert len(pairs) == 0

    def test_canonical_order(self):
        noun_lists = [["나", "가"]] * 5
        pairs = extract_cooccurrence(noun_lists, {"나", "가"}, min_count=1)
        assert ("가", "나") in pairs
        assert ("나", "가") not in pairs
