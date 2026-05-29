from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ExtractConfig(BaseModel):
    """Configuration model for the Korean ontology extractor."""

    tagger: str = Field("komoran", description="KoNLPy tagger: 'komoran' or 'mecab'")
    min_term_len: int = Field(2, ge=1, description="Minimum character length for a concept term")
    top_terms: int = Field(60, ge=1, description="Top-N terms selected by TF-IDF")
    min_freq: int = Field(2, ge=1, description="Minimum noun frequency across all sentences")
    use_cooccurrence: bool = Field(True, description="Enable co-occurrence relation extraction")
    cooc_min_count: int = Field(3, ge=1, description="Minimum co-occurrence count to create a relation")
    base_iri: str = Field(
        "http://example.org/onto#",
        description="Base IRI for the generated ontology",
    )
    output_dir: Path = Field(Path("output"), description="Directory for .owl output files")
    urls: list[str] = Field(default_factory=list, description="List of URLs to process (for run_all)")

    @field_validator("tagger")
    @classmethod
    def validate_tagger(cls, v: str) -> str:
        allowed = {"komoran", "mecab", "okt", "hannanum", "kkma"}
        if v.lower() not in allowed:
            raise ValueError(f"tagger must be one of {allowed}, got {v!r}")
        return v.lower()

    @field_validator("base_iri")
    @classmethod
    def validate_base_iri(cls, v: str) -> str:
        if not v.endswith("#") and not v.endswith("/"):
            v = v + "#"
        return v

    @classmethod
    def load(cls, yaml_path: str | Path) -> "ExtractConfig":
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**raw)

    model_config = {"extra": "forbid"}


Config = ExtractConfig
