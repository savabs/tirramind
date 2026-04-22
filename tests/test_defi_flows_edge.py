"""
Edge case tests for DefiFlowsTool (DefiLlama — on-chain DeFi data).

Covers: mode validation, limit clamping, chain/category filtering, TVL mode,
stablecoins mode, dex_volume mode, chain mode, cache interaction, HTTP errors,
timeout, empty responses, malformed data, output formatting, registry + bandit.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.tools.defi_flows import (
    DefiFlowsTool,
    VALID_MODES,
    _MARKET_CATEGORIES,
    _PROTOCOLS_URL,
    _STABLECOINS_URL,
    _DEXS_URL,
)
from agent.tools.base import ToolResult


# ── Fixtures ──────────────────────────────────────────────────


def _tool(cache=None) -> DefiFlowsTool:
    return DefiFlowsTool(cache=cache)


def _mock_resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


SAMPLE_PROTOCOLS = [
    {
        "name": "Aave",
        "tvl": 10_000_000_000,
        "chain": "Multi-Chain",
        "category": "Lending",
        "chains": ["Ethereum", "Polygon", "Avalanche"],
        "change_1d": 2.5,
        "change_7d": -1.3,
        "chainTvls": {
            "Ethereum": 7_000_000_000,
            "Polygon": 2_000_000_000,
            "Avalanche": 1_000_000_000,
        },
    },
    {
        "name": "Uniswap",
        "tvl": 5_000_000_000,
        "chain": "Multi-Chain",
        "category": "DEX",
        "chains": ["Ethereum", "Polygon", "Arbitrum"],
        "change_1d": -0.5,
        "change_7d": 3.2,
        "chainTvls": {
            "Ethereum": 4_000_000_000,
            "Polygon": 500_000_000,
            "Arbitrum": 500_000_000,
        },
    },
    {
        "name": "Lido",
        "tvl": 8_000_000_000,
        "chain": "Ethereum",
        "category": "Liquid Staking",
        "chains": ["Ethereum"],
        "change_1d": 0.1,
        "change_7d": 0.8,
        "chainTvls": {"Ethereum": 8_000_000_000},
    },
]

SAMPLE_STABLECOINS = {
    "peggedAssets": [
        {
            "name": "Tether",
            "symbol": "USDT",
            "pegMechanism": "fiat-backed",
            "circulating": {"peggedUSD": 100_000_000_000},
            "chains": ["Ethereum", "Tron"],
        },
        {
            "name": "USD Coin",
            "symbol": "USDC",
            "pegMechanism": "fiat-backed",
            "circulating": {"peggedUSD": 50_000_000_000},
            "chains": ["Ethereum", "Solana"],
        },
        {
            "name": "DAI",
            "symbol": "DAI",
            "pegMechanism": "crypto-backed",
            "circulating": {"peggedUSD": 5_000_000_000},
            "chains": ["Ethereum"],
        },
    ]
}

SAMPLE_DEXS = {
    "totalVolume": 8_000_000_000,
    "protocols": [
        {
            "name": "Uniswap",
            "totalVolume24h": 3_000_000_000,
            "change_1d": 5.2,
            "change_7d": -2.1,
            "chains": ["Ethereum", "Polygon"],
        },
        {
            "name": "PancakeSwap",
            "totalVolume24h": 1_500_000_000,
            "change_1d": -3.0,
            "change_7d": 1.5,
            "chains": ["Binance"],
        },
        {
            "name": "SushiSwap",
            "totalVolume24h": 500_000_000,
            "change_1d": 0.0,
            "change_7d": -5.0,
            "chains": ["Ethereum", "Arbitrum"],
        },
    ],
}


# ── 1. Tool Metadata ─────────────────────────────────────────


class TestToolMetadata:
    def test_name(self):
        assert _tool().name == "defi_flows"

    def test_description_nonempty(self):
        assert len(_tool().description) > 50

    def test_parameters_schema(self):
        params = _tool().parameters
        assert params["type"] == "object"
        props = params["properties"]
        assert "mode" in props
        assert "chain" in props
        assert "category" in props
        assert "limit" in props

    def test_mode_enum(self):
        modes = _tool().parameters["properties"]["mode"]["enum"]
        assert set(modes) == {"tvl", "stablecoins", "dex_volume", "chain"}

    def test_required_fields(self):
        assert _tool().parameters["required"] == ["mode"]


# ── 2. Input Validation ──────────────────────────────────────


class TestInputValidation:
    def test_invalid_mode(self):
        r = _tool().execute(mode="invalid")
        assert not r.success
        assert "Invalid mode" in r.output

    def test_empty_mode(self):
        r = _tool().execute(mode="")
        assert not r.success

    def test_no_mode(self):
        r = _tool().execute()
        assert not r.success

    def test_mode_case_sensitive(self):
        r = _tool().execute(mode="TVL")
        assert not r.success

    def test_limit_clamped_low(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", limit=0)
            assert r.success
            assert r.data["count"] >= 1

    def test_limit_clamped_high(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", limit=999)
            assert r.success
            assert r.data["count"] <= len(SAMPLE_PROTOCOLS)

    def test_limit_string_coerced(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", limit="5")
            assert r.success

    def test_extra_kwargs_ignored(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", bogus="thing")
            assert r.success


# ── 3. TVL Mode ───────────────────────────────────────────────


class TestTVLMode:
    def test_basic_tvl(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl")
            assert r.success
            assert "protocols" in r.data
            assert r.data["count"] == 3
            assert r.data["protocols"][0]["name"] == "Aave"  # highest TVL

    def test_tvl_sorted_descending(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl")
            tvls = [p["tvl_usd"] for p in r.data["protocols"]]
            assert tvls == sorted(tvls, reverse=True)

    def test_tvl_chain_filter(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", chain="Polygon")
            assert r.success
            # Only Aave and Uniswap have Polygon
            assert r.data["count"] == 2

    def test_tvl_chain_filter_case_insensitive(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", chain="polygon")
            assert r.success
            assert r.data["count"] == 2

    def test_tvl_category_filter(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", category="Lending")
            assert r.success
            assert r.data["count"] == 1
            assert r.data["protocols"][0]["name"] == "Aave"

    def test_tvl_category_filter_case_insensitive(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", category="lending")
            assert r.success
            assert r.data["count"] == 1

    def test_tvl_both_filters(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", chain="Ethereum", category="DEX")
            assert r.success
            assert r.data["count"] == 1
            assert r.data["protocols"][0]["name"] == "Uniswap"

    def test_tvl_no_match(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", chain="NonexistentChain")
            assert r.success
            assert r.data["count"] == 0

    def test_tvl_total_tvl_computed(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl")
            assert r.data["total_tvl"] == 23_000_000_000

    def test_tvl_limit_applied(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl", limit=1)
            assert r.data["count"] == 1

    def test_tvl_change_fields(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl")
            p = r.data["protocols"][0]
            assert "change_1d_pct" in p
            assert "change_7d_pct" in p

    def test_tvl_none_change(self):
        protocols = [{"name": "X", "tvl": 100, "chains": [], "category": "DEX"}]
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=protocols):
            r = _tool().execute(mode="tvl")
            assert r.success
            assert r.data["protocols"][0]["change_1d_pct"] is None

    def test_tvl_fetch_failure(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=None):
            r = _tool().execute(mode="tvl")
            assert not r.success
            assert "Failed" in r.output


# ── 4. Stablecoins Mode ──────────────────────────────────────


class TestStablecoinsMode:
    def test_basic_stablecoins(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins")
            assert r.success
            assert "stablecoins" in r.data
            assert r.data["count"] == 3
            assert r.data["stablecoins"][0]["name"] == "Tether"

    def test_stablecoins_sorted_by_supply(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins")
            supplies = [s["circulating_usd"] for s in r.data["stablecoins"]]
            assert supplies == sorted(supplies, reverse=True)

    def test_stablecoins_total_supply(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins")
            assert r.data["total_supply"] == 155_000_000_000

    def test_stablecoins_limit(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins", limit=1)
            assert r.data["count"] == 1

    def test_stablecoins_fields(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins")
            s = r.data["stablecoins"][0]
            assert "name" in s
            assert "symbol" in s
            assert "peg_mechanism" in s
            assert "circulating_usd" in s

    def test_stablecoins_fetch_failure(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=None):
            r = _tool().execute(mode="stablecoins")
            assert not r.success

    def test_stablecoins_empty(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value={"peggedAssets": []}
        ):
            r = _tool().execute(mode="stablecoins")
            assert r.success
            assert r.data["count"] == 0

    def test_stablecoins_missing_circulating(self):
        data = {"peggedAssets": [{"name": "X", "symbol": "X", "pegMechanism": "algo"}]}
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="stablecoins")
            assert r.success
            assert r.data["stablecoins"][0]["circulating_usd"] == 0


# ── 5. DEX Volume Mode ───────────────────────────────────────


class TestDEXVolumeMode:
    def test_basic_dex_volume(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume")
            assert r.success
            assert "dexes" in r.data
            assert r.data["count"] == 3
            assert r.data["dexes"][0]["name"] == "Uniswap"

    def test_dex_volume_sorted(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume")
            vols = [d["volume_24h_usd"] for d in r.data["dexes"]]
            assert vols == sorted(vols, reverse=True)

    def test_dex_volume_chain_filter(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume", chain="Binance")
            assert r.success
            assert r.data["count"] == 1
            assert r.data["dexes"][0]["name"] == "PancakeSwap"

    def test_dex_volume_total(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume")
            assert r.data["total_volume_24h"] == 8_000_000_000

    def test_dex_volume_null_volume(self):
        data = {
            "totalVolume": 0,
            "protocols": [{"name": "X", "totalVolume24h": None, "chains": []}],
        }
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="dex_volume")
            assert r.success
            assert r.data["dexes"][0]["volume_24h_usd"] is None

    def test_dex_volume_fetch_failure(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=None):
            r = _tool().execute(mode="dex_volume")
            assert not r.success

    def test_dex_volume_limit(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume", limit=2)
            assert r.data["count"] == 2


# ── 6. Chain TVL Mode ────────────────────────────────────────


class TestChainTVLMode:
    def test_basic_chain_tvl(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="chain")
            assert r.success
            assert "chains" in r.data
            # Ethereum should be top
            assert r.data["chains"][0]["chain"] == "Ethereum"

    def test_chain_tvl_sorted(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="chain")
            tvls = [c["tvl_usd"] for c in r.data["chains"]]
            assert tvls == sorted(tvls, reverse=True)

    def test_chain_tvl_protocol_count(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="chain")
            eth = next(c for c in r.data["chains"] if c["chain"] == "Ethereum")
            assert eth["protocol_count"] == 3  # All 3 have Ethereum

    def test_chain_tvl_limit(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="chain", limit=2)
            assert r.data["count"] == 2

    def test_chain_tvl_fetch_failure(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=None):
            r = _tool().execute(mode="chain")
            assert not r.success


# ── 7. HTTP Error Handling ────────────────────────────────────


class TestHTTPErrors:
    def test_timeout(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", side_effect=httpx.TimeoutException("timeout")
        ):
            r = _tool().execute(mode="tvl")
            assert not r.success
            assert "timed out" in r.output

    def test_http_error(self):
        with patch.object(
            DefiFlowsTool,
            "_fetch_json",
            side_effect=httpx.HTTPStatusError(
                "500",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            ),
        ):
            r = _tool().execute(mode="tvl")
            assert not r.success

    def test_connection_error(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", side_effect=httpx.ConnectError("fail")
        ):
            r = _tool().execute(mode="tvl")
            assert not r.success

    def test_generic_exception(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", side_effect=RuntimeError("boom")
        ):
            r = _tool().execute(mode="tvl")
            assert not r.success
            assert "Unexpected" in r.output


# ── 8. Cache Interaction ──────────────────────────────────────


class TestCacheInteraction:
    def test_cache_hit(self):
        cache = MagicMock()
        cache.get.return_value = SAMPLE_PROTOCOLS
        tool = _tool(cache=cache)
        r = tool.execute(mode="tvl")
        assert r.success
        cache.get.assert_called()

    def test_cache_miss_then_set(self):
        cache = MagicMock()
        cache.get.return_value = None
        tool = _tool(cache=cache)
        with patch("httpx.Client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_PROTOCOLS
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            r = tool.execute(mode="tvl")
            assert r.success
            cache.put.assert_called()

    def test_no_cache(self):
        tool = _tool(cache=None)
        with patch("httpx.Client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = SAMPLE_PROTOCOLS
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.__enter__ = MagicMock(
                return_value=MagicMock(get=MagicMock(return_value=mock_resp))
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            r = tool.execute(mode="tvl")
            assert r.success


# ── 9. Output Formatting ─────────────────────────────────────


class TestOutputFormatting:
    def test_tvl_output_mentions_count(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="tvl")
            assert "Top" in r.output
            assert "TVL" in r.output

    def test_stablecoins_output_mentions_supply(self):
        with patch.object(
            DefiFlowsTool, "_fetch_json", return_value=SAMPLE_STABLECOINS
        ):
            r = _tool().execute(mode="stablecoins")
            assert "supply" in r.output.lower()

    def test_dex_output_mentions_volume(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_DEXS):
            r = _tool().execute(mode="dex_volume")
            assert "volume" in r.output.lower()

    def test_chain_output_mentions_chains(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=SAMPLE_PROTOCOLS):
            r = _tool().execute(mode="chain")
            assert "chain" in r.output.lower()


# ── 10. Constants ─────────────────────────────────────────────


class TestConstants:
    def test_valid_modes(self):
        assert VALID_MODES == {"tvl", "stablecoins", "dex_volume", "chain"}

    def test_market_categories(self):
        assert "Lending" in _MARKET_CATEGORIES
        assert "DEX" in _MARKET_CATEGORIES

    def test_urls_https(self):
        assert _PROTOCOLS_URL.startswith("https://")
        assert _STABLECOINS_URL.startswith("https://")
        assert _DEXS_URL.startswith("https://")


# ── 11. Empty / Malformed Data ────────────────────────────────


class TestMalformedData:
    def test_empty_protocol_list(self):
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=[]):
            r = _tool().execute(mode="tvl")
            assert r.success
            assert r.data["count"] == 0

    def test_protocol_missing_tvl(self):
        data = [{"name": "X", "chains": ["Ethereum"], "category": "DEX"}]
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="tvl")
            assert r.success
            assert r.data["protocols"][0]["tvl_usd"] == 0

    def test_protocol_missing_chains(self):
        data = [{"name": "X", "tvl": 100, "category": "DEX"}]
        with patch.object(DefiFlowsTool, "_fetch_json", return_value=data):
            r = _tool().execute(mode="tvl", chain="Ethereum")
            assert r.success
            assert r.data["count"] == 0

    def test_dex_empty_protocols(self):
        with patch.object(
            DefiFlowsTool,
            "_fetch_json",
            return_value={"totalVolume": 0, "protocols": []},
        ):
            r = _tool().execute(mode="dex_volume")
            assert r.success
            assert r.data["count"] == 0


# ── 12. Registry + Bandit Integration ────────────────────────


class TestRegistryAndBandit:
    def test_tool_count(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert len(registry._tools) == 60

    def test_defi_flows_registered(self):
        try:
            from agent.cli import build_tool_registry
        except (ImportError, ModuleNotFoundError):
            pytest.skip("optional dep not installed")
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.tool_timeout = 30
        mock_config.fred_api_key = ""
        registry = build_tool_registry(mock_config)
        assert "defi_flows" in registry._tools

    def test_bandit_arm_count(self):
        from agent.learning.bandit import DEFAULT_ARMS

        assert len(DEFAULT_ARMS) == 48

    def test_defi_liquidity_arm_exists(self):
        from agent.learning.bandit import DEFAULT_ARMS

        names = {a.name for a in DEFAULT_ARMS}
        assert "defi_liquidity" in names

    def test_defi_arm_references_tool(self):
        from agent.learning.bandit import DEFAULT_ARMS

        arm = next(a for a in DEFAULT_ARMS if a.name == "defi_liquidity")
        assert "defi_flows" in arm.tools
