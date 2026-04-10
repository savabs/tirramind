"""
TirraMind — Entity Resolution Utilities

Deterministic entity name normalization, canonical ID generation,
and seed loaders for the entity registry.

Entity types: company, person, vessel, wallet, country, organization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

EntityType = Literal[
    "company",
    "country",
    "domain",
    "organization",
    "person",
    "protocol",
    "topic",
    "vessel",
    "wallet",
]

# Suffixes stripped during company name normalization (case-insensitive).
# Order matters: longer suffixes first to avoid partial matches.
_COMPANY_SUFFIXES = [
    r"\bincorporated\b",
    r"\bcorporation\b",
    r"\blimited\b",
    r"\bcompany\b",
    r"\bholdings?\b",
    r"\bgroup\b",
    r"\binc\b\.?",
    r"\bcorp\b\.?",
    r"\bltd\b\.?",
    r"\bllc\b\.?",
    r"\bllp\b\.?",
    r"\blp\b\.?",
    r"\bco\b\.?",
    r"\bplc\b\.?",
    r"\bsa\b",
    r"\bag\b",
    r"\bnv\b",
    r"\bse\b",
    r"/[a-z]{2}/",  # state/jurisdiction suffixes like /DE/, /NV/
]

_SUFFIX_PATTERN = re.compile(
    r"(?:" + "|".join(_COMPANY_SUFFIXES) + r")",
    re.IGNORECASE,
)


def normalize_company_name(name: str) -> str:
    """Normalize a company name to a canonical form.

    Steps:
    1. Unicode NFKD normalization (strip accents)
    2. Lowercase
    3. Strip common corporate suffixes (Inc., Corp., Ltd., etc.)
    4. Remove punctuation (except hyphens within words)
    5. Collapse whitespace
    6. Strip leading/trailing whitespace

    Returns the normalized name, or raises ValueError if the result is empty.
    """
    if not name or not name.strip():
        raise ValueError("Company name must be non-empty")

    # Unicode normalization: decompose accented characters
    normalized = unicodedata.normalize("NFKD", name)
    # Remove combining marks (accents)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Lowercase
    normalized = normalized.lower()
    # Strip corporate suffixes
    normalized = _SUFFIX_PATTERN.sub("", normalized)
    # Remove punctuation except hyphens and spaces
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        raise ValueError(f"Company name normalizes to empty string: {name!r}")

    return normalized


def entity_id_from_key(entity_type: str, key: str) -> str:
    """Generate a deterministic entity ID from type + key.

    Uses SHA-256, truncated to 16 hex chars (64 bits). Collision probability
    is negligible at entity counts < 10M.
    """
    raw = f"{entity_type}:{key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_sec_company_tickers(
    store: PipelineStore,
    json_path: str | Path | None = None,
) -> int:
    """Load SEC company_tickers.json as seed entities.

    Each entry is registered as entity_type="company" with aliases:
    - source="sec_cik", external_id=<CIK string>
    - source="ticker", external_id=<ticker>

    The canonical_name is the normalized company title.

    Args:
        store: PipelineStore instance to write to.
        json_path: Path to company_tickers.json. If None, downloads from SEC.

    Returns:
        Number of entities registered (new + already existing).
    """
    data = _load_tickers_data(json_path)
    count = 0
    for entry in data.values():
        cik = str(entry.get("cik_str", ""))
        ticker = str(entry.get("ticker", "")).upper()
        title = str(entry.get("title", ""))

        if not cik or not title:
            continue

        try:
            canonical = normalize_company_name(title)
        except ValueError:
            log.warning("Skipping SEC entity with un-normalizable name: %r", title)
            continue

        eid = entity_id_from_key("company", cik)
        store.register_entity(
            entity_type="company",
            canonical_name=canonical,
            entity_id=eid,
            metadata={"sec_title": title},
        )
        store.add_entity_alias(eid, "sec_cik", cik)
        if ticker:
            store.add_entity_alias(eid, "ticker", ticker)
        count += 1

    log.info("Loaded %d SEC company tickers as seed entities", count)
    return count


def _load_tickers_data(json_path: str | Path | None) -> dict[str, Any]:
    """Load tickers JSON from file path.

    If json_path is None, attempts to fetch from SEC EDGAR.
    """
    if json_path is not None:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"SEC tickers file not found: {path}")
        with open(path) as f:
            return json.load(f)

    # Download from SEC — uses standard urllib to avoid adding dependencies
    import urllib.request

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "TirraMind/1.0 (research@tirramind.com)"}
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))
