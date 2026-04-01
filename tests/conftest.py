"""
Shared pytest configuration and reusable fixtures for TirraMind tests.

Fixtures here are available to ALL test files automatically.
They are opt-in (request by name) — no autouse.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Markers registration (avoids "unknown marker" warnings)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that take >10s")
    config.addinivalue_line(
        "markers", "integration: marks tests requiring network or real API"
    )
    config.addinivalue_line(
        "markers", "live: marks tests hitting real external endpoints"
    )


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cache():
    """A MagicMock DataCache that always returns None (cache miss)."""
    cache = MagicMock()
    cache.get.return_value = None
    return cache


@pytest.fixture
def mock_cache_with_data():
    """Factory fixture: returns a mock cache pre-loaded with specific data.

    Usage:
        def test_foo(mock_cache_with_data):
            cache = mock_cache_with_data({"key": "value"})
    """

    def _factory(data: dict):
        cache = MagicMock()
        cache.get.side_effect = lambda key, **kw: data.get(key)
        return cache

    return _factory


# ---------------------------------------------------------------------------
# HTTP fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_httpx_response():
    """Factory: build a mock httpx.Response with configurable status/json/text.

    Usage:
        def test_foo(mock_httpx_response):
            resp = mock_httpx_response(200, json={"key": "val"})
    """

    def _factory(status_code: int = 200, *, json=None, text=None, headers=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.is_success = 200 <= status_code < 300
        resp.headers = headers or {}
        if json is not None:
            resp.json.return_value = json
            resp.text = str(json)
        elif text is not None:
            resp.text = text
            resp.json.side_effect = ValueError("not json")
        else:
            resp.json.return_value = {}
            resp.text = ""
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import httpx

            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=resp
            )
        return resp

    return _factory


# ---------------------------------------------------------------------------
# Pipeline / SQLite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path for pipeline tests."""
    return tmp_path / "test_pipeline.db"


@pytest.fixture
def memory_db():
    """In-memory SQLite connection for fast database tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Numerical fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns():
    """A small synthetic daily return series (100 days, known properties).

    Returns a plain Python list. Tests needing numpy can do np.array(sample_returns).
    """
    import random

    rng = random.Random(42)
    return [rng.gauss(0.0005, 0.02) for _ in range(100)]


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_result_success():
    """A generic successful ToolResult-like dict."""
    return {"success": True, "output": "OK", "data": {}}


@pytest.fixture
def tool_result_failure():
    """A generic failed ToolResult-like dict."""
    return {"success": False, "output": "Something went wrong", "data": {}}
