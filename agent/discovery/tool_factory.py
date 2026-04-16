"""TirraMind — Template-Based Tool Factory (Change 15)

Generates new ``Tool`` instances from discovered data source candidates.
Uses configuration-driven templates — **no runtime code generation** —
to create tools that fetch from JSON APIs or CSV feeds.

Each generated tool is serialisable to a JSON config file for persistence
across restarts.

Reference: Spec step 15.4 in [[tier8_autonomous_discovery_spec]].
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from agent.discovery.signal_evaluator import SignalReport
    from agent.discovery.source_scout import DataSourceCandidate

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = ".tirra_pipeline/discovered_tools"
_MAX_RESPONSE_BYTES = 1_048_576


class DiscoveredJsonApiTool(Tool):
    """Auto-generated tool for a JSON API data source.

    Parameters
    ----------
    source_id : str
        Unique identifier for this data source.
    source_name : str
        Human-readable name.
    url : str
        Endpoint URL.
    description : str
        What this data source provides.
    response_path : list[str]
        Path of keys to navigate to the data array in the JSON response.
        E.g., ``["result", "records"]`` for ``resp["result"]["records"]``.
    field_mapping : dict[str, str]
        Mapping from source field names to standardised output field names.
    """

    def __init__(
        self,
        source_id: str,
        source_name: str,
        url: str,
        description: str = "",
        response_path: list[str] | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        self._source_id = source_id
        self._source_name = source_name
        self._url = url
        self._description = description[:200] if description else ""
        self._response_path = response_path or []
        self._field_mapping = field_mapping or {}

    @property
    def name(self) -> str:
        return f"discovered_{self._source_id[:8]}"

    @property
    def description(self) -> str:
        return f"Auto-discovered: {self._description}" if self._description else "Auto-discovered JSON API tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "TirraMind/1.0 (research@tirramind.com)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read(_MAX_RESPONSE_BYTES)
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
            return ToolResult(success=False, output=f"Fetch failed: {exc}")

        # Navigate to data array
        try:
            target = reduce(lambda d, k: d[k], self._response_path, data)
        except (KeyError, TypeError, IndexError):
            target = data

        # Apply field mapping
        if isinstance(target, list) and self._field_mapping:
            mapped = []
            for row in target:
                if isinstance(row, dict):
                    mapped.append(
                        {
                            out_name: row.get(src_name)
                            for src_name, out_name in self._field_mapping.items()
                            if src_name in row
                        }
                    )
            target = mapped

        return ToolResult(
            success=True,
            output=f"Fetched {len(target) if isinstance(target, list) else 1} records from {self._source_name}",
            data=target,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialise tool configuration for persistence."""
        return {
            "type": "json_api",
            "source_id": self._source_id,
            "source_name": self._source_name,
            "url": self._url,
            "description": self._description,
            "response_path": self._response_path,
            "field_mapping": self._field_mapping,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiscoveredJsonApiTool:
        """Reconstruct tool from a saved config."""
        return cls(
            source_id=config["source_id"],
            source_name=config["source_name"],
            url=config["url"],
            description=config.get("description", ""),
            response_path=config.get("response_path", []),
            field_mapping=config.get("field_mapping", {}),
        )


class DiscoveredCsvFeedTool(Tool):
    """Auto-generated tool for a CSV/TSV data feed.

    Parameters
    ----------
    source_id : str
        Unique identifier.
    source_name : str
        Human-readable name.
    url : str
        Download URL.
    description : str
        Description text.
    delimiter : str
        CSV delimiter (default ',').
    field_mapping : dict[str, str]
        Source column → output column mapping.
    max_rows : int
        Maximum rows to return per fetch.
    """

    def __init__(
        self,
        source_id: str,
        source_name: str,
        url: str,
        description: str = "",
        delimiter: str = ",",
        field_mapping: dict[str, str] | None = None,
        max_rows: int = 500,
    ) -> None:
        self._source_id = source_id
        self._source_name = source_name
        self._url = url
        self._description = description[:200] if description else ""
        self._delimiter = delimiter
        self._field_mapping = field_mapping or {}
        self._max_rows = max_rows

    @property
    def name(self) -> str:
        return f"discovered_{self._source_id[:8]}"

    @property
    def description(self) -> str:
        return f"Auto-discovered: {self._description}" if self._description else "Auto-discovered CSV feed tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "TirraMind/1.0 (research@tirramind.com)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read(_MAX_RESPONSE_BYTES)
            text = raw.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ToolResult(success=False, output=f"Fetch failed: {exc}")

        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=self._delimiter)
            rows: list[dict[str, str | None]] = []
            for i, row in enumerate(reader):
                if i >= self._max_rows:
                    break
                if self._field_mapping:
                    mapped = {
                        out_name: row.get(src_name)
                        for src_name, out_name in self._field_mapping.items()
                        if src_name in row
                    }
                    rows.append(mapped)
                else:
                    rows.append(dict(row))
        except Exception as exc:
            return ToolResult(success=False, output=f"Parse failed: {exc}")

        return ToolResult(
            success=True,
            output=f"Fetched {len(rows)} rows from {self._source_name}",
            data=rows,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "csv_feed",
            "source_id": self._source_id,
            "source_name": self._source_name,
            "url": self._url,
            "description": self._description,
            "delimiter": self._delimiter,
            "field_mapping": self._field_mapping,
            "max_rows": self._max_rows,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiscoveredCsvFeedTool:
        return cls(
            source_id=config["source_id"],
            source_name=config["source_name"],
            url=config["url"],
            description=config.get("description", ""),
            delimiter=config.get("delimiter", ","),
            field_mapping=config.get("field_mapping", {}),
            max_rows=config.get("max_rows", 500),
        )


class ToolFactory:
    """Creates and persists discovered tools from candidates.

    Parameters
    ----------
    config_dir : str | Path
        Directory for persisting tool config JSON files.
    """

    def __init__(self, config_dir: str | Path = _DEFAULT_CONFIG_DIR) -> None:
        self._config_dir = Path(config_dir)

    def create_tool(
        self,
        candidate: DataSourceCandidate,
        signal_report: SignalReport,
    ) -> DiscoveredJsonApiTool | DiscoveredCsvFeedTool | None:
        """Create a tool instance from a validated candidate.

        Returns *None* if the format is unsupported.
        """
        if candidate.format == "json_api":
            response_path = self._detect_response_path(candidate.probe_sample)
            field_mapping = self._detect_field_mapping(
                candidate.probe_sample, response_path
            )
            return DiscoveredJsonApiTool(
                source_id=candidate.source_id,
                source_name=candidate.name,
                url=candidate.url,
                description=candidate.description,
                response_path=response_path,
                field_mapping=field_mapping,
            )
        elif candidate.format == "csv_feed":
            field_mapping = {}
            if isinstance(candidate.probe_sample, list) and candidate.probe_sample:
                first = candidate.probe_sample[0]
                if isinstance(first, dict):
                    field_mapping = {k: k for k in first}
            return DiscoveredCsvFeedTool(
                source_id=candidate.source_id,
                source_name=candidate.name,
                url=candidate.url,
                description=candidate.description,
                field_mapping=field_mapping,
            )
        else:
            log.warning("Unsupported format %s for %s", candidate.format, candidate.name)
            return None

    def save_config(self, tool: DiscoveredJsonApiTool | DiscoveredCsvFeedTool) -> Path:
        """Persist tool config to a JSON file."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        path = self._config_dir / f"{tool.name}.json"
        with open(path, "w") as f:
            json.dump(tool.to_config(), f, indent=2)
        log.debug("Saved tool config: %s", path)
        return path

    def load_all_configs(self) -> list[DiscoveredJsonApiTool | DiscoveredCsvFeedTool]:
        """Load all tool configs from the config directory."""
        tools: list[DiscoveredJsonApiTool | DiscoveredCsvFeedTool] = []
        if not self._config_dir.exists():
            return tools
        for path in sorted(self._config_dir.glob("*.json")):
            try:
                with open(path) as f:
                    config = json.load(f)
                tool = self._tool_from_config(config)
                if tool is not None:
                    tools.append(tool)
            except Exception:
                log.warning("Failed to load tool config: %s", path, exc_info=True)
        return tools

    @staticmethod
    def _tool_from_config(
        config: dict[str, Any],
    ) -> DiscoveredJsonApiTool | DiscoveredCsvFeedTool | None:
        tool_type = config.get("type")
        if tool_type == "json_api":
            return DiscoveredJsonApiTool.from_config(config)
        elif tool_type == "csv_feed":
            return DiscoveredCsvFeedTool.from_config(config)
        return None

    @staticmethod
    def _detect_response_path(
        probe_sample: dict | list | None,
    ) -> list[str]:
        """Heuristically detect the path to the data array in a JSON response."""
        if isinstance(probe_sample, list):
            return []  # Already a list
        if not isinstance(probe_sample, dict):
            return []
        # Common patterns: {key: [...]}
        for key in ("data", "results", "records", "rows", "items", "result"):
            val = probe_sample.get(key)
            if isinstance(val, list):
                return [key]
            if isinstance(val, dict):
                for subkey in ("data", "results", "records", "rows", "items"):
                    if isinstance(val.get(subkey), list):
                        return [key, subkey]
        return []

    @staticmethod
    def _detect_field_mapping(
        probe_sample: dict | list | None,
        response_path: list[str],
    ) -> dict[str, str]:
        """Auto-detect field mapping from probe sample."""
        if probe_sample is None:
            return {}
        # Navigate to data array
        target = probe_sample
        try:
            for key in response_path:
                target = target[key]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            pass

        rows: list[dict] = []
        if isinstance(target, list):
            rows = [r for r in target[:5] if isinstance(r, dict)]
        elif isinstance(target, dict):
            rows = [target]

        if not rows:
            return {}

        # Identity mapping — preserve all fields
        first = rows[0]
        return {k: k for k in first}
