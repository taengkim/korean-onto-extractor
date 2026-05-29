from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from onto_extractor.config import ExtractConfig
from onto_extractor.pipeline import run_all, run_one


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onto-extract",
        description="Extract a Korean OWL ontology from a web page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  onto-extract --url https://ko.wikipedia.org/wiki/인공지능
  onto-extract --config configs/example.yaml
  onto-extract --url https://... --output-dir /tmp/owl --tagger mecab
""",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", metavar="URL", help="Single URL to process")
    source.add_argument(
        "--config",
        metavar="PATH",
        type=Path,
        help="YAML config file (must contain a urls list for batch mode)",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help="Override output directory (default: output/)",
    )
    parser.add_argument(
        "--tagger",
        choices=["komoran", "mecab", "okt", "hannanum", "kkma"],
        default=None,
        help="Override morpheme tagger",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=None,
        metavar="N",
        help="Override top-N TF-IDF concept count",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logger = logging.getLogger("onto_extractor.cli")

    try:
        if args.config:
            cfg = ExtractConfig.load(args.config)
            paths = run_all(cfg)
            if not paths:
                logger.error("No URLs were processed successfully.")
                return 1
            for p in paths:
                print(p)
        else:
            overrides: dict = {}
            if args.output_dir is not None:
                overrides["output_dir"] = args.output_dir
            if args.tagger is not None:
                overrides["tagger"] = args.tagger
            if args.top_terms is not None:
                overrides["top_terms"] = args.top_terms
            cfg = ExtractConfig(**overrides)
            path = run_one(args.url, cfg)
            print(path)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except (ValueError, RuntimeError) as exc:
        logger.error("Extraction failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
