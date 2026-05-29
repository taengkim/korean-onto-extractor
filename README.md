# korean-onto-extractor

URL을 입력받아 한국어 웹 문서에서 OWL 온톨로지를 추출하는 CLI/라이브러리.
**LLM 미사용** — 규칙·통계 기반으로 결정론적이고 재현 가능합니다.

## 처리 흐름

```
URL → 본문 추출 → 문장분리/형태소 → 개념 선별 → 관계 추출 → OWL 적재
```

## 설치

```bash
pip install -e ".[dev]"
```

MeCab을 사용하려면 시스템에 `mecab-ko`도 설치해야 합니다.

## 사용법

### CLI

```bash
# 단일 URL
onto-extract --url https://ko.wikipedia.org/wiki/인공지능

# YAML 배치
onto-extract --config configs/example.yaml

# 옵션
onto-extract --url https://... --output-dir /tmp/owl --tagger mecab --top-terms 30 --verbose
```

### Python API

```python
from onto_extractor import Config, run_one, run_all

# 단일 URL
owl_path = run_one("https://...", Config(output_dir="output"))

# 설정 파일 배치
cfg = Config.load("configs/example.yaml")
paths = run_all(cfg)
```

## 설정

`configs/example.yaml` 참고:

| 키               | 기본값                      | 설명                        |
|-----------------|--------------------------|---------------------------|
| tagger          | komoran                  | 형태소 분석기 (mecab 권장)        |
| min_term_len    | 2                        | 개념 최소 글자수                 |
| top_terms       | 60                       | TF-IDF 상위 N개              |
| min_freq        | 2                        | 명사 최소 빈도                  |
| use_cooccurrence| true                     | 동시출현 관계 사용 여부             |
| cooc_min_count  | 3                        | 동시출현 임계값                  |
| base_iri        | http://example.org/onto# | 온톨로지 IRI 베이스              |

## 테스트

```bash
pytest tests/ -v
```

`test_relations.py`와 `test_terms.py`는 KoNLPy 없이 실행 가능합니다.

## 출력

`output/onto_<url해시>.owl` — RDF/XML 포맷.
Protégé로 열어 클래스 계층 및 `relatedTo` 주석을 확인할 수 있습니다.
