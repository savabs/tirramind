"""Deterministic chain brief generation for ghost pattern alerts (Phase 2.2).

Auto output is a **draft** in ``ghost_archive/briefs/draft/``.
Human-edited publish briefs live in ``ghost_archive/briefs/``.

Layer 2: Feature Engineering / surveillance output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.quant.ghost_chains import ChainTemplate, load_chain_template

DRAFT_BRIEFS_DIR = Path("ghost_archive/briefs/draft")
PUBLISH_BRIEFS_DIR = Path("ghost_archive/briefs")
_TEMPLATES_DIR = Path("templates/ghost_chains/mp1")

_MP_LABELS: dict[str, str] = {
    "MP-1": "MP-1 Atlantic Energy",
    "MP-2": "MP-2 Grain Corridor",
    "MP-3": "MP-3 Metals",
    "MP-4": "MP-4 FX Macro",
}

_READOUT_LABELS: dict[str, str] = {
    "CL=F": "WTI (`CL=F`)",
    "BZ=F": "Brent (`BZ=F`)",
    "NG=F": "Natural Gas (`NG=F`)",
}

_LAYER_PREFIX: dict[str, str] = {
    "energy_supply": "Physical layer",
    "cftc": "Positioning layer",
    "gdelt": "Geopolitical layer",
    "ais_vessel": "Flow layer",
}

_WATCH_BY_TOOL: dict[str, str] = {
    "energy_supply": "**Next EIA petroleum status report** — does the inventory signal persist?",
    "cftc": "**Friday CFTC report** — managed-money net and weekly flow on readout contracts",
    "gdelt": "**GDELT producer-country stress** — Russia / Middle East / Atlantic suppliers",
    "ais_vessel": "**Baltic / Atlantic AIS tanker activity** — routing or density shifts",
}


def _brief_seq_from_alert_id(alert_id: str) -> int:
    suffix = alert_id.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1


def next_brief_seq(briefs_dir: Path | None = None) -> int:
    """Next brief sequence number (max existing + 1 across draft and publish)."""
    seqs: list[int] = []
    for root in (briefs_dir,) if briefs_dir else (DRAFT_BRIEFS_DIR, PUBLISH_BRIEFS_DIR):
        if not root.exists():
            continue
        for path in root.glob("*_CHAIN_BRIEF_*.md"):
            m = re.search(r"_CHAIN_BRIEF_(\d+)\.md$", path.name)
            if m:
                seqs.append(int(m.group(1)))
    return max(seqs, default=0) + 1


def _format_obs_date(iso: str) -> str:
    day = iso[:10]
    try:
        y, m, d = day.split("-")
        months = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        return f"{int(d)} {months[int(m) - 1]} {y}"
    except (ValueError, IndexError):
        return day


def _z_phrase(z: float | None) -> str:
    if z is None:
        return "anomalous vs its 52-week baseline"
    az = abs(z)
    direction = "above" if z > 0 else "below"
    return f"about **{az:.1f}σ {direction}** its 52-week baseline"


def _format_value(obs: str, value: Any) -> str:
    if value is None:
        return "n/a"
    tool = obs.split("/", 1)[0] if "/" in obs else ""
    metric = obs.rsplit("/", 1)[-1] if "/" in obs else obs
    if tool == "energy_supply" and metric == "weekly_change":
        v = float(value)
        sign = "fell" if v < 0 else "rose"
        if abs(v) >= 1000:
            return f"{sign} **{abs(v) / 1000:.2f} million barrels**"
        return f"{sign} **{abs(v):,.0f} thousand barrels**"
    if tool == "energy_supply":
        v = float(value)
        if v >= 1000:
            return f"**{v / 1000:.1f} million barrels**"
        return f"**{v:,.0f} thousand barrels**"
    if tool == "cftc" and "flow" in metric:
        v = float(value)
        verb = "added" if v > 0 else "cut"
        return f"{verb} **{abs(v):,.0f} contracts** in one week"
    if tool == "cftc":
        return f"**{float(value):+,.0f} contracts** net"
    if tool == "ais_vessel":
        return f"**{float(value):,.0f}** position reports"
    if tool == "gdelt":
        return f"avg |Goldstein| **{float(value):.1f}**"
    return f"**{value}**"


def _layer_heading(obs: str) -> str:
    tool = obs.split("/", 1)[0] if "/" in obs else "signal"
    return _LAYER_PREFIX.get(tool, "Signal layer")


def _title_from_nodes(alert: dict[str, Any]) -> str:
    tools = {n["obs"].split("/", 1)[0] for n in alert.get("nodes", []) if "obs" in n}
    if "energy_supply" in tools and "cftc" in tools:
        return "Inventory Meets Positioning"
    if "gdelt" in tools and "cftc" in tools:
        return "Geopolitical Stress Meets Positioning"
    if "ais_vessel" in tools and "energy_supply" in tools:
        return "Atlantic Flow Meets Physical Tightness"
    tmpl = alert.get("chain_template", "ghost")
    return tmpl.replace("_", " ").title()


def _why_it_matters(alert: dict[str, Any], template: ChainTemplate | None) -> str:
    n_domains = len({n["obs"].split("/", 1)[0] for n in alert.get("nodes", [])})
    readout = alert.get("readout_instrument", "CL=F")
    desc = (template.description if template else "").strip()
    lead = (
        f"This alert links **{n_domains} domains** on the Atlantic energy book "
        f"with readout **{readout}**."
    )
    if desc:
        lead = desc.split("\n")[0].strip()
    return (
        f"{lead}\n\n"
        "When physical, flow, and positioning layers move together against their own "
        "histories, the ghost pattern is the **combination** — not any single indicator. "
        "This is not a buy signal. It is a **scenario flag**: tighten hedges and "
        "liquidity plans before the crowd reprices.\n"
    )


def _what_to_watch(nodes: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for node in nodes:
        tool = node.get("obs", "").split("/", 1)[0]
        if tool in seen or tool not in _WATCH_BY_TOOL:
            continue
        seen.add(tool)
        items.append(_WATCH_BY_TOOL[tool])
    if not items:
        items.append("**Readout instrument** forward path over the next 2–5 sessions")
    return items


def _outcome_footer(alert: dict[str, Any]) -> str:
    outcome = alert.get("outcome")
    if outcome:
        direction = outcome.get("direction", "?")
        ret = outcome.get("return_pct", 0)
        notes = outcome.get("notes", "")
        return (
            f"*Resolved: {alert.get('readout_instrument', 'CL=F')} "
            f"**{ret:+.2f}%** ({direction}). {notes}*"
        )
    return (
        "*Outcome: pending — run `python scripts/resolve_ghost_alert.py --all` "
        "after 2–5 trading sessions.*"
    )


def alert_to_brief_markdown(
    alert: dict[str, Any],
    template: ChainTemplate | None = None,
    *,
    brief_seq: int | None = None,
) -> str:
    """Render a full chain brief markdown document from an alert dict."""
    seq = brief_seq if brief_seq is not None else _brief_seq_from_alert_id(alert["alert_id"])
    title_suffix = _title_from_nodes(alert)
    title = f"Chain Brief #{seq} — {title_suffix}"
    issued = alert.get("issued_at", "")[:10]
    mp = alert.get("micro_playground", "MP-1")
    mp_label = _MP_LABELS.get(mp, mp)
    readout = _READOUT_LABELS.get(
        alert.get("readout_instrument", "CL=F"),
        alert.get("readout_instrument", "CL=F"),
    )
    tmpl_id = alert.get("chain_template", "unknown")
    alert_rel = f"ghost_archive/alerts/{alert['alert_id']}.json"

    happened: list[str] = []
    for node in alert.get("nodes", []):
        layer = _layer_heading(node.get("obs", ""))
        entity = node.get("entity", "Unknown")
        date_phrase = _format_obs_date(node.get("observed_at", ""))
        val_phrase = _format_value(node.get("obs", ""), node.get("value"))
        z_phrase = _z_phrase(node.get("z"))
        happened.append(
            f"**{layer}:** {entity} registered {val_phrase} "
            f"({z_phrase}) as of **{date_phrase}**."
        )

    watch_lines = "\n".join(
        f"{i}. {line}" for i, line in enumerate(_what_to_watch(alert.get("nodes", [])), 1)
    )

    source_rows = []
    for node in alert.get("nodes", []):
        url = node.get("source_url", "")
        label = node.get("entity", node.get("obs", "source"))
        date_s = node.get("observed_at", "")[:10]
        source_rows.append(f"| {label} | [{url}]({url}) — {date_s} |")

    score = alert.get("chain_score", 0)
    method = (
        f"**DRAFT** — auto-generated by `ghost_pattern_scan.py` "
        f"(chain_score = {score}). Edit into `ghost_archive/briefs/` before publishing. "
        f"Z-scores from 52-week rolling history in `pipeline.db`."
    )
    happened_text = "\n\n".join(happened)
    sources_text = "\n".join(source_rows)

    body = f"""---
title: "{title}"
tags:
  - doc/memory
  - topic/ghost-patterns
  - topic/commercial
  - status/active
  - layer/surveillance
---

# {title}

**Micro-playground:** {mp_label}  
**Issued:** {issued}  
**Readout:** {readout}  
**Template:** `{tmpl_id}`  
**Alert:** `{alert_rel}`

---

## What happened

{happened_text}

## Why it matters

{_why_it_matters(alert, template)}

## What to watch next

{watch_lines}

## Method note

{method}

## Sources

| Node | Source |
|------|--------|
{sources_text}

---

{_outcome_footer(alert)}

## Related

- [[ghost_pattern_income_plan]]
- [[ghost_pattern_income_task]]
- [[scorecard]]
"""
    return body


def _load_template_for_alert(alert: dict[str, Any]) -> ChainTemplate | None:
    tmpl_id = alert.get("chain_template")
    if not tmpl_id:
        return None
    path = _TEMPLATES_DIR / f"{tmpl_id}.yaml"
    if not path.exists():
        return None
    return load_chain_template(path)


def brief_path_for_alert(
    alert: dict[str, Any],
    briefs_dir: Path | None = None,
    *,
    draft: bool = True,
) -> Path:
    """Canonical brief path for an alert (matches alert sequence suffix)."""
    if briefs_dir is None:
        briefs_dir = DRAFT_BRIEFS_DIR if draft else PUBLISH_BRIEFS_DIR
    seq = _brief_seq_from_alert_id(alert["alert_id"])
    issued = alert.get("issued_at", alert["alert_id"][:10])[:10]
    mp = alert.get("micro_playground", "MP-1")
    return briefs_dir / f"{issued}_{mp}_CHAIN_BRIEF_{seq:03d}.md"


def write_brief_for_alert(
    alert: dict[str, Any],
    *,
    briefs_dir: Path | None = None,
    template: ChainTemplate | None = None,
    overwrite: bool = False,
    draft: bool = True,
) -> Path:
    """Write draft chain brief markdown for an alert. Returns brief path."""
    if briefs_dir is None:
        briefs_dir = DRAFT_BRIEFS_DIR if draft else PUBLISH_BRIEFS_DIR
    briefs_dir.mkdir(parents=True, exist_ok=True)
    out = brief_path_for_alert(alert, briefs_dir)
    if out.exists() and not overwrite:
        return out
    if template is None:
        template = _load_template_for_alert(alert)
    md = alert_to_brief_markdown(alert, template)
    out.write_text(md, encoding="utf-8")
    return out


def _patch_outcome_footer(path: Path, footer: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\*Outcome:.*?\*", footer, text, count=1, flags=re.DOTALL)
    text = re.sub(r"\*Resolved:.*?\*", footer, text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def update_brief_outcome(alert: dict[str, Any]) -> list[Path]:
    """Refresh outcome footer on draft and publish briefs after resolution."""
    alert_rel = f"ghost_archive/alerts/{alert['alert_id']}.json"
    footer = _outcome_footer(alert)
    updated: list[Path] = []

    for briefs_dir, draft in ((DRAFT_BRIEFS_DIR, True), (PUBLISH_BRIEFS_DIR, False)):
        if not briefs_dir.exists():
            continue
        path = brief_path_for_alert(alert, briefs_dir, draft=draft)
        if not path.exists():
            for candidate in briefs_dir.glob("*.md"):
                try:
                    if alert_rel in candidate.read_text(encoding="utf-8"):
                        path = candidate
                        break
                except OSError:
                    continue
            else:
                continue
        _patch_outcome_footer(path, footer)
        updated.append(path)
    return updated
