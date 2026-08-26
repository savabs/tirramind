"""Layer-1 vendor-contract regression tests (agent/tools/).

Each test here pins a contract that was *observed* against the live vendor API,
not inferred from an error message:

* FEC ``/candidates/search/`` rejects ``sort=-election_year`` with HTTP 422 and
  names ``election_years`` (plural) in the error body.
* UN Comtrade's public *preview* endpoint returns ``reporterDesc``,
  ``partnerDesc``, ``cmdDesc`` and ``flowDesc`` as explicit ``null`` on every
  row, and needs ``cmdCode=TOTAL`` to produce a per-partner breakdown.
* ``DataCache`` exposes only ``get(source, params)`` / ``put(source, params,
  data)`` — no ``.set()``, no ``ttl=`` kwarg.
"""

from __future__ import annotations

import ast
import json
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.data.cache import DataCache
from agent.tools import comtrade as comtrade_mod
from agent.tools.comtrade import ComtradeTool, _parse_trade_records
from agent.tools.political_risk import PoliticalRiskTool


@pytest.fixture(autouse=True)
def _offline_area_names(monkeypatch):
    """Pin Comtrade's M49 area table so no test touches the network."""
    monkeypatch.setattr(
        comtrade_mod,
        "_AREA_NAMES",
        {0: "World", 124: "Canada", 156: "China", 484: "Mexico", 842: "USA"},
        raising=False,
    )
    yield


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1] / "agent" / "tools"


def _resp(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=json.dumps(body),
        request=httpx.Request("GET", "http://test"),
    )


def _patched_client(response: httpx.Response):
    """Patch httpx.Client and expose the recorded .get() call."""
    mc = patch("httpx.Client")
    m = mc.start()
    m.return_value.__enter__ = lambda s: s
    m.return_value.__exit__ = MagicMock(return_value=False)
    m.return_value.get.return_value = response
    return mc, m


# ═══════════════════════════════════════════════════════════════════════
# FEC — /candidates/search/ sort key
# ═══════════════════════════════════════════════════════════════════════


class TestFECCandidateSort:
    def test_candidates_does_not_sort_on_singular_election_year(self):
        """`sort=-election_year` is a 422 from the live API; must be plural.

        Live evidence (2026-08-26, DEMO_KEY):
            GET /v1/candidates/search/?sort=-election_year  -> HTTP 422
              'Cannot sort on value "-election_year". Instead choose one of:
               ... "election_years" ...'
            GET /v1/candidates/search/?sort=-election_years -> HTTP 200
        """
        mc, m = _patched_client(_resp({"results": [], "pagination": {"count": 0}}))
        try:
            PoliticalRiskTool(cache=None).execute(mode="candidates", query="Trump")
            _, kwargs = m.return_value.get.call_args
        finally:
            mc.stop()

        sort = kwargs["params"]["sort"]
        assert sort == "-election_years", f"FEC rejects {sort!r} with HTTP 422"
        assert sort != "-election_year"

    def test_422_surfaces_the_api_message(self):
        """A validation error must name the offending parameter, not hide it."""
        body = {
            "message": 'Cannot sort on value "-election_year". Instead choose one of: "election_years"',
            "status": 422,
        }
        mc, m = _patched_client(_resp(body, status=422))
        try:
            r = PoliticalRiskTool(cache=None).execute(mode="candidates")
        finally:
            mc.stop()

        assert r.success is False
        assert "election_years" in r.output


# ═══════════════════════════════════════════════════════════════════════
# UN Comtrade — preview endpoint sends null labels on every row
# ═══════════════════════════════════════════════════════════════════════


def _preview_row(partner_code: int, value: float) -> dict:
    """Verbatim shape of a public-preview row (all *Desc fields null)."""
    return {
        "period": "2024",
        "reporterCode": 842,
        "reporterISO": None,
        "reporterDesc": None,
        "flowCode": "X",
        "flowDesc": None,
        "partnerCode": partner_code,
        "partnerISO": None,
        "partnerDesc": None,
        "cmdCode": "TOTAL",
        "cmdDesc": None,
        "aggrLevel": None,
        "primaryValue": value,
        "qty": 0.0,
        "qtUnit": None,
    }


class TestComtradePreviewNullLabels:
    def test_null_partner_desc_is_resolved_from_the_numeric_code(self):
        """`.get(k, default)` never fires on an explicit null — codes must win.

        Regression: every row rendered as "World (TOTAL)" because partnerDesc
        was null and the formatter defaulted it to World.
        """
        data = {"data": [_preview_row(124, 3.0), _preview_row(484, 2.0), _preview_row(156, 1.0)]}
        recs = _parse_trade_records(data)

        assert len(recs) == 3
        labels = [r["partner"] for r in recs]
        assert len(set(labels)) == 3, f"partners collapsed to {labels}"
        assert not any(str(x).startswith("World") for x in labels), labels
        assert [r["partner_code"] for r in recs] == [124, 484, 156]

    def test_world_row_is_labelled_world_only_when_code_is_zero(self):
        recs = _parse_trade_records({"data": [_preview_row(0, 9.0)]})
        assert recs[0]["partner"] == "World"

    def test_null_flow_desc_falls_back_to_the_flow_code(self):
        recs = _parse_trade_records({"data": [_preview_row(124, 1.0)]})
        assert recs[0]["flow"] == "Export"
        assert recs[0]["flow_code"] == "X"

    def test_missing_keys_still_yield_unknown(self):
        """Absent (not null) codes must not crash or invent a country."""
        recs = _parse_trade_records({"data": [{"period": "2024"}]})
        assert recs[0]["reporter"] == "Unknown"
        assert recs[0]["partner"] == "Unknown"

    def test_partners_mode_requests_cmdcode_total(self):
        """Without cmdCode=TOTAL the API returns a partner x HS-chapter grid
        that the preview endpoint truncates at 500 rows, so most partners are
        missing entirely.  Live: with TOTAL, USA/2024/X -> 224 partner rows."""
        mc, m = _patched_client(_resp({"count": 0, "data": []}))
        try:
            ComtradeTool(cache=None).execute(mode="partners", reporter="USA", period="2024")
            _, kwargs = m.return_value.get.call_args
        finally:
            mc.stop()

        params = kwargs.get("params", kwargs)
        assert params.get("cmdCode") == "TOTAL", params
        assert params.get("reporterCode") == "842"

    def test_every_query_pins_the_sub_aggregate_dimensions(self):
        """Reporters that publish customs/mode-of-transport breakdowns return
        the full cross-product unless these are pinned.

        Live evidence (2026-08-26), DEU / 2024 / X / cmdCode=TOTAL:
            unpinned -> 500 rows (preview cap), 197 distinct partners,
                        duplicate USA rows, "World" = $375,319,898,683
            pinned   -> 229 rows, 229 distinct partners, 0 duplicates,
                        "World" = $1,630,712,507,953
        """
        for mode, extra in (
            ("partners", {}),
            ("flows", {"partner": "CHN"}),
            ("commodity", {"commodity_code": "8542"}),
        ):
            mc, m = _patched_client(_resp({"count": 0, "data": []}))
            try:
                ComtradeTool(cache=None).execute(mode=mode, reporter="USA", period="2024", **extra)
                _, kwargs = m.return_value.get.call_args
            finally:
                mc.stop()
            params = kwargs.get("params", kwargs)
            assert params.get("customsCode") == "C00", (mode, params)
            assert params.get("motCode") == "0", (mode, params)
            assert params.get("partner2Code") == "0", (mode, params)

    def test_partners_output_does_not_repeat_one_label(self):
        rows = [_preview_row(0, 100.0), _preview_row(124, 30.0), _preview_row(484, 20.0)]
        mc, m = _patched_client(_resp({"count": 3, "data": rows}))
        try:
            r = ComtradeTool(cache=None).execute(mode="partners", reporter="USA", period="2024")
        finally:
            mc.stop()

        assert r.success
        assert r.data["record_count"] == 3
        assert r.output.count("World") == 1, r.output


# ═══════════════════════════════════════════════════════════════════════
# DataCache API conformance across every Layer-1 tool
# ═══════════════════════════════════════════════════════════════════════


class TestCacheApiConformance:
    def test_datacache_surface_is_get_and_put_only(self):
        assert not hasattr(DataCache, "set")
        import inspect

        put_params = list(inspect.signature(DataCache.put).parameters)
        assert put_params == ["self", "source", "params", "data"]

    def test_no_tool_calls_a_nonexistent_cache_api(self):
        """18 tools once called cache.set()/put(ttl=); every fetch was lost on
        save and the mocked tests still passed.  Keep it at zero."""
        offenders: list[str] = []
        for path in sorted(TOOLS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(), str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                recv = ast.unparse(node.func.value).lower()
                if "cache" not in recv or recv.endswith("cached"):
                    continue
                if node.func.attr == "set":
                    offenders.append(f"{path.name}:{node.lineno} cache.set()")
                elif node.func.attr == "put":
                    kw = [k.arg for k in node.keywords]
                    if "ttl" in kw or len(node.args) != 3:
                        offenders.append(f"{path.name}:{node.lineno} put(args={len(node.args)}, kw={kw})")
        assert offenders == [], offenders
