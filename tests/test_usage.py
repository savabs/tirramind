"""Tests for the usage-metering store (per-subscriber API call log)."""

from __future__ import annotations

import time

import pytest

from agent.payments.usage import UsageStore


@pytest.fixture
def store(tmp_path):
    return UsageStore(str(tmp_path / "usage.db"))


def test_log_and_count_since(store):
    store.log(key_id="k1", endpoint="/api/v1/data", tier="data")
    store.log(key_id="k1", endpoint="/api/v1/data", tier="data")
    store.log(key_id="k2", endpoint="/api/v1/data", tier="data")

    assert store.count_since("k1", since=0.0) == 2
    assert store.count_since("k2", since=0.0) == 1
    assert store.count_since("k3", since=0.0) == 0


def test_count_since_respects_time_window(store):
    store.log(key_id="k1", endpoint="/api/v1/data")
    future = time.time() + 10
    assert store.count_since("k1", since=future) == 0


def test_usage_by_endpoint_groups_correctly(store):
    store.log(key_id="k1", endpoint="/api/v1/data")
    store.log(key_id="k1", endpoint="/api/v1/data")
    store.log(key_id="k1", endpoint="/evidence/stats")

    by_endpoint = store.usage_by_endpoint("k1")
    assert by_endpoint == {"/api/v1/data": 2, "/evidence/stats": 1}


def test_usage_by_endpoint_isolated_per_key(store):
    store.log(key_id="k1", endpoint="/api/v1/data")
    store.log(key_id="k2", endpoint="/api/v1/data")
    store.log(key_id="k2", endpoint="/api/v1/data")

    assert store.usage_by_endpoint("k1") == {"/api/v1/data": 1}
    assert store.usage_by_endpoint("k2") == {"/api/v1/data": 2}


def test_summary_combines_total_and_breakdown(store):
    store.log(key_id="k1", endpoint="/api/v1/data")
    store.log(key_id="k1", endpoint="/api/v1/sources")
    store.log(key_id="k1", endpoint="/api/v1/sources")

    summary = store.summary("k1")
    assert summary["total"] == 3
    assert summary["by_endpoint"] == {"/api/v1/data": 1, "/api/v1/sources": 2}


def test_summary_empty_for_unknown_key(store):
    summary = store.summary("nobody")
    assert summary == {"total": 0, "by_endpoint": {}}
