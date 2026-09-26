"""Tests for canonical country-code resolution.

Guards the entity-identity failure documented in
``docs/research/graph_connectivity_failure.md``: GDELT wrote countries as CAMEO
alpha-3 (``USA``) while the instrument universe wrote them as ISO alpha-2
(``US``), so the same country became two entities and the world-events graph
could not reach a single instrument.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.pipeline.country_codes import (
    CAMEO_REGION_CODES,
    ISO_ALPHA2_NAMES,
    ISO_ALPHA3_TO_ALPHA2,
    country_name,
    is_region_code,
    resolve_country_key,
)

_LIVE_DB = Path(__file__).resolve().parents[1] / ".tirra_pipeline" / "pipeline.db"


class TestResolveCountryKey:
    @pytest.mark.parametrize(
        ("alpha3", "alpha2"),
        [
            ("USA", "US"),
            ("FRA", "FR"),
            ("LBN", "LB"),
            ("CAN", "CA"),
            ("JPN", "JP"),
            ("RUS", "RU"),
            ("DEU", "DE"),
            ("PRK", "KP"),  # alpha-3 initial letter differs from alpha-2
            ("COM", "KM"),  # Comoros: no shared prefix
            ("CHE", "CH"),
        ],
    )
    def test_alpha3_maps_to_alpha2(self, alpha3: str, alpha2: str) -> None:
        assert resolve_country_key(alpha3) == alpha2

    @pytest.mark.parametrize("alpha2", ["US", "FR", "LB", "CA", "JP", "KP"])
    def test_alpha2_passes_through(self, alpha2: str) -> None:
        assert resolve_country_key(alpha2) == alpha2

    def test_idempotent(self) -> None:
        """resolve(resolve(x)) == resolve(x) for every known code."""
        for code in list(ISO_ALPHA3_TO_ALPHA2) + list(ISO_ALPHA2_NAMES):
            once = resolve_country_key(code)
            assert once is not None
            assert resolve_country_key(once) == once

    def test_case_and_whitespace_insensitive(self) -> None:
        assert resolve_country_key(" usa ") == "US"
        assert resolve_country_key("Usa") == "US"

    @pytest.mark.parametrize("region", ["EUR", "AFR", "MEA", "SEA", "CRB", "LAM", "WST", "SAM"])
    def test_regional_blocs_resolve_to_none(self, region: str) -> None:
        """CAMEO bloc codes are not countries and must never become one."""
        assert resolve_country_key(region) is None
        assert is_region_code(region)

    def test_legacy_codes(self) -> None:
        assert resolve_country_key("TMP") == "TL"  # East Timor -> Timor-Leste
        assert resolve_country_key("ZAR") == "CD"  # Zaire -> DR Congo

    @pytest.mark.parametrize("bad", [None, "", "   ", "ZZZ", "X", "12345"])
    def test_unknown_returns_none(self, bad: str | None) -> None:
        assert resolve_country_key(bad) is None

    def test_none_is_not_coerced_to_placeholder(self) -> None:
        """A None return means 'write no country entity' — never a fallback."""
        assert resolve_country_key("EUR") is None
        assert resolve_country_key("EUR") != "EU"


class TestCountryName:
    def test_replaces_actor_name_garbage(self) -> None:
        """GDELT's Actor1Name put ALASKA / SASKATCHEWAN / TOYOTA in as countries."""
        assert country_name("US") == "United States"
        assert country_name("CA") == "Canada"
        assert country_name("JP") == "Japan"
        assert country_name("FR") == "France"

    def test_unknown_returns_none(self) -> None:
        assert country_name("ZZ") is None
        assert country_name(None) is None

    def test_every_mapped_alpha2_has_a_name(self) -> None:
        missing = sorted(set(ISO_ALPHA3_TO_ALPHA2.values()) - set(ISO_ALPHA2_NAMES))
        assert not missing, f"alpha-2 codes without a display name: {missing}"


class TestTableIntegrity:
    def test_no_alpha3_collides_with_a_region_code(self) -> None:
        assert not (set(ISO_ALPHA3_TO_ALPHA2) & CAMEO_REGION_CODES)

    def test_alpha2_values_are_unique(self) -> None:
        values = list(ISO_ALPHA3_TO_ALPHA2.values())
        dupes = {v for v in values if values.count(v) > 1}
        assert not dupes, f"alpha-2 codes mapped from multiple alpha-3: {dupes}"

    def test_codes_are_well_formed(self) -> None:
        for a3, a2 in ISO_ALPHA3_TO_ALPHA2.items():
            assert len(a3) == 3 and a3.isupper(), a3
            assert len(a2) == 2 and a2.isupper(), a2


@pytest.mark.skipif(not _LIVE_DB.exists(), reason="live pipeline.db not present")
class TestLiveGraphCoverage:
    """Every country code in the live graph must resolve or be a known region.

    This fails loudly if a collector starts emitting an unmapped code, which is
    the condition that silently split the graph in the first place.
    """

    @staticmethod
    def _live_codes() -> list[str]:
        con = sqlite3.connect(f"file:{_LIVE_DB}?mode=ro", uri=True)
        try:
            return [
                row[0]
                for row in con.execute(
                    "SELECT a.external_id FROM entity_aliases a "
                    "JOIN entities e ON e.entity_id = a.entity_id "
                    "WHERE e.entity_type = 'country' AND a.source = 'fips'"
                )
            ]
        finally:
            con.close()

    def test_all_live_codes_are_classified(self) -> None:
        unclassified = sorted(
            {code for code in self._live_codes() if resolve_country_key(code) is None and not is_region_code(code)}
        )
        assert not unclassified, (
            f"{len(unclassified)} country codes in the live graph neither resolve "
            f"to ISO alpha-2 nor are known CAMEO regions: {unclassified}. "
            "Add them to ISO_ALPHA3_TO_ALPHA2, LEGACY_ALPHA3, or "
            "CAMEO_REGION_CODES before they split the graph again."
        )
