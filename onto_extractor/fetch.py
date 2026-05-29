from __future__ import annotations

import logging

import trafilatura

logger = logging.getLogger(__name__)


def fetch_text(url: str) -> str:
    """Fetch body text from a URL using trafilatura.

    Args:
        url: The web page URL to fetch.

    Returns:
        Extracted plain text (body only, ads/menus stripped).

    Raises:
        ValueError: If trafilatura cannot extract any text.
        RuntimeError: If the download itself fails.
    """
    logger.debug("Fetching URL: %s", url)

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise RuntimeError(f"Failed to download content from URL: {url!r}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
    )

    if not text or not text.strip():
        raise ValueError(f"No extractable body text found at URL: {url!r}")

    logger.info("Fetched %d characters from %s", len(text), url)
    return text.strip()
