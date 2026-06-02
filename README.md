# korean-onto-extractor

URL을 입력받아 한국어 웹 문서에서 OWL 온톨로지를 추출하는 CLI/라이브러리.  
**LLM 미사용** — 규칙·통계 기반으로 결정론적이고 재현 가능합니다.

---

## 목차

1. [처리 흐름](#처리-흐름)
2. [디렉터리 구조](#디렉터리-구조)
3. [설치](#설치)
4. [사용법](#사용법)
5. [설정 항목](#설정-항목)
6. [사전 시스템](#사전-시스템)
   - [개체 사전](#개체-사전-entitiesyaml)
   - [트리거 사전](#트리거-사전-triggersyaml)
   - [단위 정규화 사전](#단위-정규화-사전-unitsyaml)
   - [관계 트리거 사전](#관계-트리거-사전-relation_triggersyaml)
   - [불용어·노이즈 사전](#불용어노이즈-사전-stopwordsyaml)
   - [커스텀 사전 적용](#커스텀-사전-적용)
7. [출력 형식](#출력-형식)
8. [테스트](#테스트)
9. [품질 조정 가이드](#품질-조정-가이드)

---

## 처리 흐름

```
URL
 │
 ▼
fetch          trafilatura로 광고·메뉴 제거 후 본문 추출
 │
 ▼
split_sentences  한국어 종결어미 기반 문장 분리
 │
 ▼
extract_nouns    KoNLPy(Komoran/MeCab)로 명사 추출
 │              └─ use_phrases=True: .pos()로 바이그램 복합명사도 추출
 │                 예) [인공, 지능] → 인공지능 추가
 ▼
normalize_nouns  단위 변형 정규화 + 노이즈 제거  ◄─ units / stopwords 사전
 │
 ▼
select_terms     TF-IDF 개념 선별 + 불용어 필터 + 개체 주입  ◄─ entities 사전
 │
 ├─► extract_isa            한국어 Hearst 패턴 (10종) → is-a 쌍
 ├─► extract_typed_relations 관계 트리거 패턴 → (주어, 관계유형, 목적어)  ◄─ relation_triggers 사전
 └─► extract_cooccurrence   문장 내 동시출현 → related_to 쌍
 │
 ▼
build_owl        OWL/RDF-XML 생성 (subClassOf · ObjectProperty · AnnotationProperty)
```

---

## 디렉터리 구조

```
korean-onto-extractor/
├── pyproject.toml
├── README.md
├── configs/
│   └── example.yaml              # 실행 설정 예시
├── onto_extractor/
│   ├── __init__.py
│   ├── config.py                 # ExtractConfig (Pydantic v2)
│   ├── fetch.py                  # fetch_text(url) → str
│   ├── preprocess.py             # split_sentences / extract_nouns / normalize_nouns
│   ├── terms.py                  # select_terms()
│   ├── relations.py              # extract_isa / extract_typed_relations / extract_cooccurrence
│   ├── owl_builder.py            # build_owl()
│   ├── pipeline.py               # run_one / run_all
│   ├── cli.py                    # onto-extract 엔트리포인트
│   ├── lexicon.py                # Lexicon 로더 클래스
│   └── lexicons/                 # 패키지 내장 사전 (YAML)
│       ├── entities.yaml         # 개체 사전
│       ├── triggers.yaml         # 트리거 사전
│       ├── units.yaml            # 단위 정규화 사전
│       ├── relation_triggers.yaml# 관계 트리거 사전
│       └── stopwords.yaml        # 불용어·노이즈 사전
└── tests/
    ├── test_relations.py         # Hearst 패턴 + 동시출현 단위 테스트
    ├── test_terms.py             # select_terms 단위 테스트
    ├── test_lexicon.py           # Lexicon 로딩 및 통합 테스트
    └── test_preprocess.py        # 명사구 추출 단위 테스트
```

---

## 설치

```bash
# 편집 가능 모드 (개발)
pip install -e ".[dev]"
```

**시스템 의존성 (선택)**

| 형태소 분석기 | 설치 명령 | 비고 |
|-------------|-----------|------|
| Komoran | (자동, 순수 Java) | 기본값, 추가 설치 불필요 |
| MeCab | `apt install mecab mecab-ko mecab-ko-dic` | 속도 우수, 정확도 높음 |

---

## 사용법

### CLI

```bash
# 단일 URL
onto-extract --url https://ko.wikipedia.org/wiki/인공지능

# YAML 배치
onto-extract --config configs/example.yaml

# 주요 옵션 오버라이드
onto-extract --url https://... \
  --output-dir /tmp/owl \
  --tagger mecab \
  --top-terms 30 \
  --verbose
```

### Python API

```python
from onto_extractor import Config, run_one, run_all

# 단일 URL (기본 설정)
owl_path = run_one("https://ko.wikipedia.org/wiki/인공지능", Config())

# 커스텀 설정
cfg = Config(
    tagger="mecab",
    top_terms=40,
    min_freq=3,
    output_dir="output",
    lexicon_dir=None,   # None = 패키지 내장 사전 사용
)
owl_path = run_one("https://...", cfg)

# 설정 파일 배치 실행
cfg = Config.load("configs/example.yaml")
paths = run_all(cfg)
```

---

## 설정 항목

`configs/example.yaml` 또는 `Config(...)` 생성자로 지정합니다.

| 키 | 기본값 | 설명 |
|---|---|---|
| 키 | 기본값 | 설명 |
|---|---|---|
| `tagger` | `komoran` | 형태소 분석기: `komoran` · `mecab` · `okt` · `hannanum` · `kkma` |
| `min_term_len` | `2` | 개념 최소 글자수 |
| `top_terms` | `60` | TF-IDF 상위 N개 개념 |
| `min_freq` | `2` | 명사 최소 출현 빈도 |
| `use_phrases` | `true` | 복합명사 추출 여부 (인공+지능→인공지능) |
| `use_cooccurrence` | `true` | 동시출현 관계(`relatedTo`) 사용 여부 |
| `cooc_min_count` | `3` | 동시출현 임계값 |
| `base_iri` | `http://example.org/onto#` | 온톨로지 IRI 베이스 |
| `output_dir` | `output` | .owl 파일 출력 디렉터리 |
| `lexicon_dir` | `null` | 사전 디렉터리 경로 (`null` = 패키지 내장 사전) |
| `urls` | `[]` | 배치 처리 URL 목록 (`run_all` 용) |

**청킹 룰 설정** (문장 분리·명사구 추출 규칙을 도메인에 맞게 조정)

| 키 | 기본값 | 설명 |
|---|---|---|
| `sent_endings` | `["다","요","죠","군요"]` | 문장 경계로 인식할 한국어 어미 목록 |
| `min_sent_len` | `5` | 분리 후 유지할 문장의 최소 길이(글자수) |
| `phrase_max_len` | `2` | 복합명사구 최대 형태소 수 (2=바이그램, 3=트라이그램까지) |

---

## 사전 시스템

패키지에 포함된 5종 YAML 사전(`onto_extractor/lexicons/`)이 파이프라인 각 단계에서 동작합니다.  
사전 파일은 도메인에 맞게 직접 편집하거나 커스텀 경로로 교체할 수 있습니다.

---

### 개체 사전 (`entities.yaml`)

**역할**: 도메인 핵심 개체를 카테고리별로 정의합니다.  
TF-IDF 점수가 낮아 `top_terms` 컷에서 탈락하더라도 `min_freq`를 만족하면 개념 목록에 강제 포함됩니다.

**형식**

```yaml
카테고리명:
  - 개체1
  - 개체2
  - 개체3
```

**예시**

```yaml
# 기존 카테고리 확장
기술:
  - 인공지능
  - 기계학습
  - 트랜스포머       # 추가
  - 거대언어모델     # 추가

# 새 카테고리 추가 (예: 법률 도메인)
법률:
  - 민법
  - 형법
  - 행정법
  - 헌법
  - 계약
  - 불법행위
  - 손해배상
```

**동작 방식**

```
select_terms() 호출 시:
  전역 빈도 ≥ min_freq 이고 entities.yaml에 있는 단어
    → TF-IDF 점수 무관하게 개념 후보에 추가
```

---

### 트리거 사전 (`triggers.yaml`)

**역할**: 특정 관계 유형을 암시하는 단어 목록입니다.  
`Lexicon.is_trigger_sentence(sent)` 및 `trigger_types(sent)` API로 문장을 분류하는 데 사용됩니다.

**형식**

```yaml
관계유형_이름:
  - 트리거단어1
  - 트리거단어2
```

**예시**

```yaml
# 기존 is_a 트리거 확장
is_a:
  - 일종
  - 유형
  - 해당된다
  - 속한다
  - 분류된다    # 추가
  - 범주에든다  # 추가

# 새 관계 유형 추가
temporal:
  - 이전에
  - 이후에
  - 동시에
  - 선행한다
  - 후속된다
```

**활용 예시 (파이썬)**

```python
from onto_extractor.lexicon import Lexicon

lexicon = Lexicon.load()

sent = "신경망은 딥러닝의 일종이다."
print(lexicon.is_trigger_sentence(sent))   # True
print(lexicon.trigger_types(sent))         # ['is_a']
```

---

### 단위 정규화 사전 (`units.yaml`)

**역할**: 형태소 분석 후 나오는 단위 변형 표현을 표준형으로 통일합니다.  
`normalize_nouns()` 단계에서 명사 단위를 치환합니다.

**형식**

```yaml
표준형:
  - 변형1
  - 변형2
  - 변형3
```

**예시**

```yaml
# 기존 항목 확장
킬로그램:
  - kg
  - 킬로
  - kilogram
  - 킬로그람    # 추가

# 의학 단위 추가
밀리몰:
  - mmol
  - 밀리몰농도

국제단위:
  - IU
  - iu
  - 아이유

# 금융 단위 추가
원:
  - KRW
  - krw
  - 한국원

달러:
  - USD
  - usd
  - 미달러
  - "$"
```

**주의**: 키(표준형)와 값(변형)의 비교는 **소문자** 기준입니다 (`kg`과 `KG` 모두 매칭됩니다).

---

### 관계 트리거 사전 (`relation_triggers.yaml`)

**역할**: 한국어 문장 패턴으로 **유형 있는 관계**(typed relation)를 추출합니다.  
각 패턴이 매칭되면 OWL `ObjectProperty`가 생성되고, `SomeValuesFrom` 제약으로 클래스에 부착됩니다.

**형식**

```yaml
관계유형:          # OWL ObjectProperty 이름이 됩니다
  - pattern: |-    # 정규식 (re.UNICODE 컴파일)
      (?P<subj>[가-힣]+)패턴(?P<obj>[가-힣]+)
    swapped: false # true 이면 subj/obj 역할을 뒤집습니다
```

**필드 설명**

| 필드 | 설명 |
|------|------|
| `pattern` | Python `re` 정규식. `(?P<subj>...)` 와 `(?P<obj>...)` 그룹 필수 |
| `swapped` | 문법상 목적어가 먼저 나오는 패턴일 때 `true`로 지정 |

**예시**

```yaml
# 새 관계 유형: 발견자 관계
discoveredBy:
  - pattern: |-
      (?P<subj>[가-힣]+)(?:은|는|이|가)\s+(?P<obj>[가-힣]+)에\s+의해\s+발견
    swapped: false
  - pattern: |-
      (?P<obj>[가-힣]+)(?:이|가)\s+(?P<subj>[가-힣]+)을\s+발견
    swapped: true  # "과학자가 원소를 발견" → subj=원소, obj=과학자 → swapped

# 새 관계 유형: 치료 관계 (의학 도메인)
treats:
  - pattern: |-
      (?P<subj>[가-힣]+)(?:은|는|이|가)\s+(?P<obj>[가-힣]+)을\s+(?:치료|완화)
    swapped: false
  - pattern: |-
      (?P<obj>[가-힣]+)\s+치료에\s+사용되는\s+(?P<subj>[가-힣]+)
    swapped: true

# 패턴 작성 팁:
# - [가-힣]+  : 한국어 음절 1개 이상 (조사 포함해서 매칭됨)
# - (?:는|은|이|가) : 조사 옵션
# - \s+       : 공백 1개 이상 (반드시 이중 이스케이프 불필요 — YAML 블록 스칼라 사용)
# - swapped: true 사용 시, (?P<subj>...)를 의미론적 주어(도메인 쪽)에 쓰세요
```

**OWL 출력 결과**

패턴이 `("컴퓨터", "hasPart", "반도체")` 트리플을 추출하면:

```xml
<owl:Class rdf:about="#%EC%BB%B4%ED%93%A8%ED%84%B0">
  <rdfs:subClassOf>
    <owl:Restriction>
      <owl:onProperty rdf:resource="#hasPart"/>
      <owl:someValuesFrom rdf:resource="#%EB%B0%98%EB%8F%84%EC%B2%B4"/>
    </owl:Restriction>
  </rdfs:subClassOf>
</owl:Class>
```

---

### 불용어·노이즈 사전 (`stopwords.yaml`)

**역할**:  
- `stopwords`: 형태소 분석 결과에서 도메인 개념이 아닌 명사를 제거합니다.  
- `noise_patterns`: 정규식으로 숫자·연도·단순 코드 등 노이즈를 제거합니다.

**형식**

```yaml
stopwords:
  - 단어1
  - 단어2

noise_patterns:
  - '정규식1'   # re.fullmatch 사용
  - '정규식2'
```

**예시**

```yaml
# 도메인별 불용어 추가
stopwords:
  # 기존 불용어 유지 ...

  # 논문·학술 도메인 추가
  - 제안
  - 실험
  - 비교
  - 분석
  - 평가
  - 검증
  - 연구
  - 논문
  - 학습

  # 뉴스 도메인 추가
  - 기자
  - 특파원
  - 보도
  - 취재
  - 발표
  - 성명

# 노이즈 패턴 추가
noise_patterns:
  - '^\d+$'         # 순수 숫자
  - '^\d{4}년$'     # 연도
  - '^[A-Z]{2,5}$'  # 대문자 약어 (예: GDP, WHO) - 필요 시 제거
  - '^\d+화$'       # 회차 (예: 3화, 12화)
  - '^제\d+$'       # 제N호 패턴
```

**패턴 작성 규칙**

| 패턴 | 매칭 예시 | 설명 |
|------|----------|------|
| `^\d+$` | `42`, `100` | 순수 숫자 |
| `^\d{4}년$` | `2024년` | 연도 |
| `^[가-힣]\d+$` | `표1`, `그림3` | 글자+숫자 |
| `^\d+[가-힣]{1,2}$` | `3번`, `15개` | 숫자+단위 |
| `^[A-Za-z]$` | `A`, `b` | 단일 알파벳 |

---

### 커스텀 사전 적용

도메인 특화 사전을 별도 디렉터리에 만들어 적용하는 방법입니다.

**1. 디렉터리 준비**

```bash
mkdir -p my_lexicons
# 5개 파일 중 필요한 것만 작성 — 없는 파일은 경고 후 스킵됩니다
cp onto_extractor/lexicons/*.yaml my_lexicons/   # 기본값 복사 후 수정
```

**2. CLI 적용**

```bash
# configs/my_domain.yaml
# lexicon_dir: my_lexicons
onto-extract --config configs/my_domain.yaml
```

**3. Python API 적용**

```python
from onto_extractor import Config, run_one

cfg = Config(
    lexicon_dir="my_lexicons",  # 커스텀 사전 경로
    tagger="mecab",
    top_terms=50,
)
owl_path = run_one("https://...", cfg)
```

**4. 런타임에서 Lexicon 직접 사용**

```python
from onto_extractor.lexicon import Lexicon

lexicon = Lexicon.load("my_lexicons")

# 단어 조회
print(lexicon.is_stopword("것"))         # True
print(lexicon.is_entity("인공지능"))     # True
print(lexicon.entity_category("물리학")) # '과학'
print(lexicon.normalize("kg"))           # '킬로그램'
print(lexicon.is_noise("2024년"))        # True
print(lexicon.is_trigger_sentence(
    "신경망은 딥러닝의 일종이다."))       # True
```

---

## 출력 형식

**파일명**: `output/onto_<URL-SHA1-12자리>.owl`  
**포맷**: RDF/XML (OWL 2)

| 추출 결과 | OWL 표현 |
|----------|---------|
| 개념 | `owl:Class` (rdfs:label = 원본 한국어) |
| is-a 쌍 (Hearst) | `rdfs:subClassOf` |
| 유형 있는 관계 (관계 트리거) | `ObjectProperty.someValuesFrom` 제약 |
| 동시출현 (relatedTo) | `AnnotationProperty relatedTo` |

Protégé에서 열어 클래스 계층, ObjectProperty, 주석을 확인할 수 있습니다.

---

## 테스트

```bash
# 전체 테스트 (KoNLPy 없이 실행 가능)
pytest tests/ -v

# 특정 파일만
pytest tests/test_relations.py -v   # Hearst 패턴
pytest tests/test_terms.py -v       # TF-IDF 선별
pytest tests/test_lexicon.py -v     # 사전 로딩 및 통합
```

| 테스트 파일 | 내용 | KoNLPy 필요 |
|------------|------|------------|
| `test_relations.py` | Hearst 패턴 10종 + 동시출현 | 불필요 |
| `test_terms.py` | 빈도·TF-IDF 선별, 결정론성 | 불필요 |
| `test_lexicon.py` | 사전 로딩, 불용어·노이즈·단위·개체·관계트리거 | 불필요 |

---

## 품질 조정 가이드

| 문제 | 조정 방법 |
|------|---------|
| 관계 추출이 너무 적다 | `relation_triggers.yaml`에 패턴 추가, `min_freq` 낮춤 |
| 노이즈 개념이 많다 | `stopwords.yaml`에 불용어 추가, `min_freq` 높임 |
| 도메인 핵심어가 빠진다 | `entities.yaml`에 개체 추가 |
| 단위 표현이 제각각이다 | `units.yaml`에 변형 추가 |
| 동시출현 노이즈가 많다 | `cooc_min_count` 높이거나 `use_cooccurrence: false` |
| 속도가 느리다 | `tagger: mecab`으로 변경, `top_terms` 줄임 |
