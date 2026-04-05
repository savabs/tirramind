"""
Unit tests for agent.security.tool_policy — tool risk tiers, SSRF protection,
path sandboxing, policy guard, and phase-based tool filtering.
"""

from __future__ import annotations

import pytest

from agent.security.tool_policy import (
    OrchestratorPhase,
    ToolPolicyGuard,
    ToolRisk,
    TrustLevel,
    get_tool_risk,
    get_tools_for_phase,
    get_trust_level_for_tool,
    is_safe_path,
    is_safe_url,
    set_allowed_roots,
)


# =====================================================================
# Tool Risk Classification
# =====================================================================


class TestToolRisk:
    def test_known_read_only(self):
        assert get_tool_risk("market_data") == ToolRisk.READ_ONLY
        assert get_tool_risk("macro_data") == ToolRisk.READ_ONLY
        assert get_tool_risk("gdelt") == ToolRisk.READ_ONLY
        assert get_tool_risk("list_directory") == ToolRisk.READ_ONLY
        assert get_tool_risk("read_file") == ToolRisk.READ_ONLY

    def test_data_fetch(self):
        assert get_tool_risk("web_search") == ToolRisk.DATA_FETCH
        assert get_tool_risk("web_browse") == ToolRisk.DATA_FETCH

    def test_state_changing(self):
        assert get_tool_risk("write_file") == ToolRisk.STATE_CHANGING

    def test_dangerous(self):
        assert get_tool_risk("run_shell") == ToolRisk.DANGEROUS
        assert get_tool_risk("execute_python") == ToolRisk.DANGEROUS

    def test_unknown_defaults_to_state_changing(self):
        assert get_tool_risk("some_future_tool_xyz") == ToolRisk.STATE_CHANGING

    def test_all_57_tools_classified(self):
        """Every tool registered in cli.py should have an explicit risk classification."""
        from agent.security.tool_policy import TOOL_RISK_REGISTRY

        assert len(TOOL_RISK_REGISTRY) >= 57


class TestTrustLevel:
    def test_web_tools_untrusted(self):
        assert get_trust_level_for_tool("web_search") == TrustLevel.TOOL_UNTRUSTED
        assert get_trust_level_for_tool("web_browse") == TrustLevel.TOOL_UNTRUSTED

    def test_api_tools_trusted(self):
        assert get_trust_level_for_tool("market_data") == TrustLevel.TOOL_TRUSTED
        assert get_trust_level_for_tool("gdelt") == TrustLevel.TOOL_TRUSTED
        assert get_trust_level_for_tool("insider_filings") == TrustLevel.TOOL_TRUSTED

    def test_dangerous_tools_untrusted(self):
        # Shell/python output depends on what was executed — conservatively untrusted
        assert get_trust_level_for_tool("run_shell") == TrustLevel.TOOL_UNTRUSTED
        assert get_trust_level_for_tool("execute_python") == TrustLevel.TOOL_UNTRUSTED


# =====================================================================
# SSRF Protection
# =====================================================================


class TestSSRFProtection:
    """is_safe_url() must block private IPs, metadata endpoints, and dangerous schemes."""

    # --- Should BLOCK ---

    def test_block_localhost(self):
        safe, reason = is_safe_url("http://localhost/admin")
        assert not safe
        assert "localhost" in reason.lower()

    def test_block_127_0_0_1(self):
        safe, _ = is_safe_url("http://127.0.0.1/")
        assert not safe

    def test_block_127_variant(self):
        safe, _ = is_safe_url("http://127.0.0.2/")
        assert not safe

    def test_block_10_network(self):
        safe, _ = is_safe_url("http://10.0.0.1/internal")
        assert not safe

    def test_block_172_16(self):
        safe, _ = is_safe_url("http://172.16.0.1/")
        assert not safe

    def test_block_192_168(self):
        safe, _ = is_safe_url("http://192.168.1.1/")
        assert not safe

    def test_block_metadata_ip(self):
        """AWS/GCP/Azure metadata endpoint."""
        safe, _ = is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert not safe

    def test_block_metadata_hostname(self):
        safe, _ = is_safe_url("http://metadata.google.internal/computeMetadata/v1/")
        assert not safe

    def test_block_ipv6_loopback(self):
        safe, _ = is_safe_url("http://[::1]/")
        assert not safe

    def test_block_file_scheme(self):
        safe, reason = is_safe_url("file:///etc/passwd")
        assert not safe
        assert "scheme" in reason.lower()

    def test_block_ftp_scheme(self):
        safe, _ = is_safe_url("ftp://evil.com/payload")
        assert not safe

    def test_block_empty_hostname(self):
        safe, _ = is_safe_url("http:///path")
        assert not safe

    def test_block_credentials_in_url(self):
        safe, reason = is_safe_url("http://user:pass@example.com/")
        assert not safe
        assert "credential" in reason.lower()

    def test_block_zero_ip(self):
        safe, _ = is_safe_url("http://0.0.0.0/")
        assert not safe

    def test_block_decimal_ip_127001(self):
        """2130706433 == 127.0.0.1 in decimal notation."""
        safe, _ = is_safe_url("http://2130706433/")
        assert not safe

    def test_block_hex_ip_loopback(self):
        """0x7f000001 == 127.0.0.1 in hex."""
        safe, _ = is_safe_url("http://0x7f000001/")
        assert not safe

    def test_block_carrier_grade_nat(self):
        safe, _ = is_safe_url("http://100.64.0.1/")
        assert not safe

    # --- Should ALLOW ---

    def test_allow_normal_https(self):
        safe, _ = is_safe_url("https://api.example.com/data")
        assert safe

    def test_allow_normal_http(self):
        safe, _ = is_safe_url("http://www.example.com/page")
        assert safe

    def test_allow_public_ip(self):
        safe, _ = is_safe_url("http://8.8.8.8/")
        assert safe

    def test_allow_port_80(self):
        safe, _ = is_safe_url("http://example.com:80/")
        assert safe

    def test_allow_port_443(self):
        safe, _ = is_safe_url("https://example.com:443/")
        assert safe

    def test_allow_port_8080(self):
        safe, _ = is_safe_url("http://example.com:8080/")
        assert safe

    def test_allow_known_api_urls(self):
        """Real URLs used by TirraMind data tools."""
        urls = [
            "https://api.stlouisfed.org/fred/series/observations",
            "https://efts.sec.gov/LATEST/search-index",
            "https://html.duckduckgo.com/html/",
            "https://gamma-api.polymarket.com/events",
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            "https://api.fda.gov/drug/drugsfda.json",
        ]
        for url in urls:
            safe, reason = is_safe_url(url)
            assert safe, f"Legitimate URL blocked: {url} — {reason}"

    def test_malformed_url(self):
        safe, reason = is_safe_url("not a url at all")
        assert not safe


# =====================================================================
# Path Sandboxing
# =====================================================================


class TestPathSandboxing:
    def test_no_roots_allows_all_with_warning(self):
        """Backward compat: if no roots configured, everything is allowed."""
        safe, _ = is_safe_path("/etc/passwd", allowed_roots=[])
        # When empty list passed explicitly, no roots → allow with warning
        assert safe

    def test_allowed_path(self, tmp_path):
        target = str(tmp_path / "data" / "file.txt")
        safe, _ = is_safe_path(target, allowed_roots=[str(tmp_path)])
        assert safe

    def test_blocked_path(self, tmp_path):
        safe, _ = is_safe_path("/etc/shadow", allowed_roots=[str(tmp_path)])
        assert not safe

    def test_traversal_attack(self, tmp_path):
        evil = str(tmp_path / ".." / ".." / "etc" / "passwd")
        safe, _ = is_safe_path(evil, allowed_roots=[str(tmp_path)])
        assert not safe

    def test_home_expansion(self, tmp_path):
        """~ expansion should be resolved before checking."""
        safe, _ = is_safe_path("~/secret", allowed_roots=[str(tmp_path)])
        assert not safe

    def test_multiple_roots(self, tmp_path):
        root1 = str(tmp_path / "workspace")
        root2 = str(tmp_path / "cache")
        (tmp_path / "workspace").mkdir()
        (tmp_path / "cache").mkdir()
        safe1, _ = is_safe_path(
            str(tmp_path / "workspace" / "f.py"), allowed_roots=[root1, root2]
        )
        safe2, _ = is_safe_path(
            str(tmp_path / "cache" / "data.json"), allowed_roots=[root1, root2]
        )
        safe3, _ = is_safe_path("/root/.ssh/id_rsa", allowed_roots=[root1, root2])
        assert safe1
        assert safe2
        assert not safe3


# =====================================================================
# Policy Guard
# =====================================================================


class TestPolicyGuard:
    def test_read_only_always_allowed(self):
        guard = ToolPolicyGuard(autonomous_mode=True)
        ok, _ = guard.check_execution("market_data", {"symbol": "AAPL"})
        assert ok

    def test_dangerous_blocked_in_autonomous(self):
        guard = ToolPolicyGuard(autonomous_mode=True)
        ok, reason = guard.check_execution("run_shell", {"command": "ls"})
        assert not ok
        assert "BLOCKED" in reason

    def test_dangerous_allowed_when_not_autonomous(self):
        guard = ToolPolicyGuard(autonomous_mode=False)
        ok, _ = guard.check_execution("run_shell", {"command": "ls"})
        assert ok

    def test_dangerous_allowed_in_whitelist(self):
        guard = ToolPolicyGuard(
            autonomous_mode=True,
            allowed_dangerous=frozenset({"execute_python"}),
        )
        ok, _ = guard.check_execution("execute_python", {"code": "print(1)"})
        assert ok
        # But run_shell is still blocked
        ok2, _ = guard.check_execution("run_shell", {"command": "ls"})
        assert not ok2

    def test_ssrf_blocked_for_data_fetch(self):
        guard = ToolPolicyGuard(autonomous_mode=False)
        ok, reason = guard.check_execution(
            "web_browse", {"url": "http://169.254.169.254/"}
        )
        assert not ok
        assert "SSRF" in reason

    def test_ssrf_allowed_for_legit_url(self):
        guard = ToolPolicyGuard(autonomous_mode=False)
        ok, _ = guard.check_execution("web_browse", {"url": "https://example.com/"})
        assert ok

    def test_path_sandbox_enforced(self, tmp_path):
        guard = ToolPolicyGuard(
            autonomous_mode=False,
            allowed_roots=[str(tmp_path)],
        )
        ok, _ = guard.check_execution("write_file", {"path": str(tmp_path / "ok.txt")})
        assert ok
        ok2, reason = guard.check_execution("write_file", {"path": "/etc/passwd"})
        assert not ok2
        assert "sandbox" in reason.lower()

    def test_state_changing_warned_in_autonomous(self):
        guard = ToolPolicyGuard(autonomous_mode=True)
        # Should be allowed but logged (we just test it doesn't block)
        ok, _ = guard.check_execution("write_file", {"path": "/tmp/test.txt"})
        assert ok  # allowed (no path roots configured → permissive backward compat)

    def test_unknown_tool_treated_as_state_changing(self):
        guard = ToolPolicyGuard(autonomous_mode=True)
        # Unknown tool = STATE_CHANGING = warned but allowed
        ok, _ = guard.check_execution("some_new_tool", {"x": 1})
        assert ok

    def test_autonomous_mode_setter(self):
        guard = ToolPolicyGuard(autonomous_mode=False)
        assert not guard.autonomous_mode
        guard.autonomous_mode = True
        assert guard.autonomous_mode
        ok, _ = guard.check_execution("run_shell", {"command": "whoami"})
        assert not ok


# =====================================================================
# Phase-based Tool Filtering
# =====================================================================


class TestPhaseFiltering:
    ALL_TOOLS = [
        "market_data",
        "web_search",
        "web_browse",
        "write_file",
        "run_shell",
        "execute_python",
        "read_file",
    ]

    def test_research_phase(self):
        tools = get_tools_for_phase(self.ALL_TOOLS, OrchestratorPhase.RESEARCH)
        assert "market_data" in tools
        assert "web_search" in tools
        assert "web_browse" in tools
        assert "read_file" in tools
        assert "write_file" not in tools
        assert "run_shell" not in tools
        assert "execute_python" not in tools

    def test_planning_phase(self):
        tools = get_tools_for_phase(self.ALL_TOOLS, OrchestratorPhase.PLANNING)
        assert "market_data" in tools
        assert "read_file" in tools
        assert "web_search" not in tools
        assert "web_browse" not in tools
        assert "run_shell" not in tools

    def test_execution_phase(self):
        tools = get_tools_for_phase(self.ALL_TOOLS, OrchestratorPhase.EXECUTION)
        # Execution allows everything
        for t in self.ALL_TOOLS:
            assert t in tools

    def test_synthesis_phase(self):
        tools = get_tools_for_phase(self.ALL_TOOLS, OrchestratorPhase.SYNTHESIS)
        assert "market_data" in tools
        assert "read_file" in tools
        assert "web_browse" not in tools
        assert "run_shell" not in tools


# =====================================================================
# ToolResult Trust Labels (integration with base.py)
# =====================================================================


class TestToolResultTrust:
    def test_default_trust_level(self):
        from agent.tools.base import ToolResult

        r = ToolResult(success=True, output="ok")
        assert r.trust_level == "tool_trusted"

    def test_custom_trust_level(self):
        from agent.tools.base import ToolResult

        r = ToolResult(success=True, output="ok", trust_level="tool_untrusted")
        assert r.trust_level == "tool_untrusted"


# =====================================================================
# Fact Taint Tracking (integration with store.py)
# =====================================================================


class TestFactTaint:
    def test_default_not_tainted(self):
        from agent.memory.store import Fact

        f = Fact(key="k", content="c", source="s", confidence=0.5)
        assert f.tainted is False

    def test_tainted_flag(self):
        from agent.memory.store import Fact

        f = Fact(
            key="k", content="c", source="web_search", confidence=0.5, tainted=True
        )
        assert f.tainted is True
