"""Korean OWL ontology extractor — public API."""

from onto_extractor.config import Config, ExtractConfig
from onto_extractor.pipeline import run_all, run_one

__all__ = ["Config", "ExtractConfig", "run_one", "run_all"]
