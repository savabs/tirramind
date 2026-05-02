"""Phase 47 — _backfill bypass tests (2 per tool × 6 tools = 12 tests).

Verifies:
  - _backfill=True skips the days_back clamp (bypass path)
  - _backfill=False (default) still applies the clamp (guard path)
"""

from __future__ import annotations

from unittest.mock import patch

from agent.tools.base import ToolResult
from agent.tools.disease_surveillance import DiseaseSurveillanceTool
from agent.tools.earthquake_proximity import EarthquakeProximityTool
from agent.tools.form144 import Form144Tool
from agent.tools.insider_filings import InsiderFilingsTool
from agent.tools.internet_infrastructure import InternetInfrastructureTool
from agent.tools.sanctions_monitor import SanctionsMonitorTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK = ToolResult(success=True, output="ok", data={})


# ---------------------------------------------------------------------------
# 1. EarthquakeProximityTool  (clamp: min(days_back, 30))
# ---------------------------------------------------------------------------


class TestEarthquakeBypass:
    def test_bypass_removes_clamp(self):
        tool = EarthquakeProximityTool()
        with patch.object(tool, "_execute_recent", return_value=_OK) as mock:
            tool.execute(days_back=1825, _backfill=True)
            assert mock.call_args.kwargs["days_back"] == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = EarthquakeProximityTool()
        with patch.object(tool, "_execute_recent", return_value=_OK) as mock:
            tool.execute(days_back=1825)
            assert mock.call_args.kwargs["days_back"] == 30


# ---------------------------------------------------------------------------
# 2. DiseaseSurveillanceTool  (clamp: min(days_back, 180))
# ---------------------------------------------------------------------------


class TestDiseaseBypass:
    def test_bypass_removes_clamp(self):
        tool = DiseaseSurveillanceTool()
        with patch.object(tool, "_execute_wastewater", return_value=_OK) as mock:
            tool.execute(mode="wastewater", days_back=1825, _backfill=True)
            assert mock.call_args.kwargs["days_back"] == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = DiseaseSurveillanceTool()
        with patch.object(tool, "_execute_wastewater", return_value=_OK) as mock:
            tool.execute(mode="wastewater", days_back=1825)
            assert mock.call_args.kwargs["days_back"] == 180


# ---------------------------------------------------------------------------
# 3. InsiderFilingsTool  (clamp: min(days_back, 90))
# ---------------------------------------------------------------------------


class TestInsiderFilingsBypass:
    def test_bypass_removes_clamp(self):
        tool = InsiderFilingsTool()
        with patch.object(tool, "_fetch_recent_filings", return_value=[]) as mock:
            tool.execute(days_back=1825, _backfill=True)
            start_dt, end_dt = mock.call_args.args
            assert (end_dt - start_dt).days == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = InsiderFilingsTool()
        with patch.object(tool, "_fetch_recent_filings", return_value=[]) as mock:
            tool.execute(days_back=1825)
            start_dt, end_dt = mock.call_args.args
            assert (end_dt - start_dt).days == 90


# ---------------------------------------------------------------------------
# 4. Form144Tool  (clamp: min(days_back, 60))
# ---------------------------------------------------------------------------


class TestForm144Bypass:
    def test_bypass_removes_clamp(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]) as mock:
            tool.execute(days_back=1825, _backfill=True)
            start_dt, end_dt = mock.call_args.args
            assert (end_dt - start_dt).days == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = Form144Tool()
        with patch.object(tool, "_fetch_recent_144s", return_value=[]) as mock:
            tool.execute(days_back=1825)
            start_dt, end_dt = mock.call_args.args
            assert (end_dt - start_dt).days == 60


# ---------------------------------------------------------------------------
# 5. SanctionsMonitorTool  (clamp: min(days_back, 365))
# ---------------------------------------------------------------------------


class TestSanctionsBypass:
    def test_bypass_removes_clamp(self):
        tool = SanctionsMonitorTool()
        with patch.object(tool, "_execute_recent", return_value=_OK) as mock:
            tool.execute(mode="recent", days_back=1825, _backfill=True)
            assert mock.call_args.kwargs["days_back"] == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = SanctionsMonitorTool()
        with patch.object(tool, "_execute_recent", return_value=_OK) as mock:
            tool.execute(mode="recent", days_back=1825)
            assert mock.call_args.kwargs["days_back"] == 365


# ---------------------------------------------------------------------------
# 6. InternetInfrastructureTool  (clamp: min(days_back, 90), used by censorship mode)
# ---------------------------------------------------------------------------


class TestInternetInfraBypass:
    def test_bypass_removes_clamp(self):
        tool = InternetInfrastructureTool()
        with patch.object(tool, "_execute_censorship", return_value=_OK) as mock:
            tool.execute(
                mode="censorship",
                country="US",
                days_back=1825,
                _backfill=True,
            )
            assert mock.call_args.kwargs["days_back"] == 1825

    def test_clamp_still_applies_without_flag(self):
        tool = InternetInfrastructureTool()
        with patch.object(tool, "_execute_censorship", return_value=_OK) as mock:
            tool.execute(mode="censorship", country="US", days_back=1825)
            assert mock.call_args.kwargs["days_back"] == 90
