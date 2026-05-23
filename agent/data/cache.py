"""
TirraMind — Local Data Cache

Saves fetched data to disk as JSON. Avoids re-downloading data that's
still fresh. Cache key = deterministic hash of (source, params).

Cache layout:
    .tirra_cache/
        <hex_key>.json    →  {"fetched_at": iso_ts, "params": {...}, "data": ...}
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone; UTC = timezone.utc
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = ".tirra_cache"
_DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours


class DataCache:
    """Simple JSON file cache keyed by deterministic hash of request params."""

    def __init__(
        self,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._dir = Path(cache_dir)
        self._ttl = ttl_seconds

    def get(self, source: str, params: dict[str, Any]) -> Any | None:
        """Return cached data if fresh, else None."""
        path = self._path_for(source, params)
        if not path.exists():
            return None

        try:
            entry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt cache entry %s — ignoring", path.name)
            return None

        fetched_at = entry.get("fetched_at", 0)
        age = time.time() - fetched_at
        if age > self._ttl:
            log.debug("Cache expired for %s (age=%.0fs, ttl=%ds)", path.name, age, self._ttl)
            return None

        log.debug("Cache hit: %s (age=%.0fs)", path.name, age)
        return entry.get("data")

    def put(self, source: str, params: dict[str, Any], data: Any) -> None:
        """Write data to cache."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(source, params)
        entry = {
            "fetched_at": time.time(),
            "fetched_at_human": datetime.now(UTC).isoformat(),
            "source": source,
            "params": params,
            "data": data,
        }
        try:
            path.write_text(json.dumps(entry, default=str))
            log.debug("Cached: %s → %s", source, path.name)
        except OSError as exc:
            log.warning("Failed to write cache: %s", exc)

    def invalidate(self, source: str, params: dict[str, Any]) -> None:
        """Remove a specific cache entry."""
        path = self._path_for(source, params)
        path.unlink(missing_ok=True)

    def clear(self) -> int:
        """Remove all cache files. Returns count of files removed."""
        if not self._dir.exists():
            return 0
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink()
            count += 1
        return count

    def _path_for(self, source: str, params: dict[str, Any]) -> Path:
        """Deterministic file path for a given source + params combo."""
        key_str = json.dumps({"source": source, "params": params}, sort_keys=True)
        digest = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return self._dir / f"{digest}.json"
