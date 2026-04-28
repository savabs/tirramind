#!/usr/bin/env python3
"""GNN Attention Diagnostic — post-Phase-40 analysis of entity coverage gaps.

Loads the trained HetTGN model and runs the full attention diagnostic:
  • Per-entity-type density and observation counts
  • Per-edge-type mean HGT attention weights (identifies starved connections)
  • Neighborhood sparsity per entity type
  • Ranked list of data-starved entity types and which tools to wire next

Usage:
    python scripts/gnn_attention_diagnostic.py [--db-path PATH] [--model-path PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from agent.models.gnn.integration import compute_diagnostics, format_diagnostic_report
from agent.models.gnn.trainer import Trainer, TrainerConfig
from agent.pipeline.store import PipelineStore

console = Console()
log = logging.getLogger(__name__)


# Tier 1 unwired tools most likely to fill data gaps per entity type
_TOOL_SUGGESTIONS: dict[str, list[str]] = {
    "company": [
        "fetch_bankruptcy_court (L2 — company entities)",
        "fetch_foia_requests (L2 — company/organization entities)",
        "fetch_academic_preprints (L2 — topic/company entities)",
    ],
    "organization": [
        "fetch_regulatory_gazette (L2 — organization entities, low volume today)",
        "fetch_gov_contracts (already wired — confirm org entity creation)",
    ],
    "instrument": [
        "fetch_polymarket (L2 — topic/instrument entities)",
        "fetch_dark_pool (L2 — instrument entities, Phase 6e)",
        "fetch_clinical_trials (L2 — company/topic entities → instrument links)",
    ],
    "person": [
        "fetch_insider_filings (already L2 — confirm person entity volume)",
        "fetch_form144 (already L2 — confirm person entity volume)",
    ],
    "vessel": [
        "fetch_ais_vessel (already wired — check obs accumulation)",
        "fetch_ship_detention (potential new L2 tool)",
    ],
    "wallet": [
        "fetch_whale_alert (already L2 — low historical volume, expected)",
        "fetch_defi_flows (already L2 — low historical volume, expected)",
    ],
    "cftc_contract": [
        "fetch_cftc (already wired — backfill to 2011 complete, check obs)",
    ],
    "topic": [
        "fetch_gdelt (already wired — 10yr backfill complete, 900K+ obs)",
        "fetch_academic_preprints (L2 — research_velocity on topic nodes)",
    ],
    "country": [
        "fetch_sovereign_debt (already L2 — sovereign_yield obs)",
        "fetch_capital_flows (already L2 — capital_flow obs)",
        "fetch_global_pmi (already L2 — economic_activity obs)",
    ],
    "domain": [
        "fetch_cert_transparency (already wired — domain entities)",
        "fetch_dns_monitor (already wired — domain entities)",
    ],
    "protocol": [
        "fetch_defi_flows (already L2 — protocol entities)",
    ],
}

# Attention threshold below which an edge type is considered data-starved
_ATTENTION_LOW_THRESHOLD = 0.03


def _print_attention_table(report: dict) -> None:
    """Print edge-type attention rankings."""
    section = report.get("edge_attention", {})
    edge_attn = section.get("values", {})
    if not edge_attn:
        console.print("[yellow]No attention data — model may not have been loaded.[/]")
        return

    flagged_edges = set(section.get("flagged", {}).keys())

    tbl = Table(title="Edge-Type HGT Attention (ranked highest → lowest)")
    tbl.add_column("Edge Type", style="cyan")
    tbl.add_column("Mean Attention", justify="right")
    tbl.add_column("Status")

    items = sorted(edge_attn.items(), key=lambda x: -x[1])
    for edge_type, attn in items:
        status = "[red]STARVED[/]" if edge_type in flagged_edges else "[green]ok[/]"
        tbl.add_row(edge_type, f"{attn:.4f}", status)
    console.print(tbl)


def _print_entity_density_table(report: dict) -> None:
    """Print entity-type density rankings."""
    section = report.get("entity_density", {})
    density = section.get("values", {})
    flagged_entities = set(section.get("flagged", {}).keys())

    tbl = Table(title="Entity-Type Density")
    tbl.add_column("Entity Type", style="green")
    tbl.add_column("Entity Count", justify="right")
    tbl.add_column("Status")

    for etype, count in sorted(density.items(), key=lambda x: -x[1]):
        status = "[red]SPARSE[/]" if etype in flagged_entities else "[green]ok[/]"
        tbl.add_row(etype, str(count), status)
    console.print(tbl)


def _print_obs_density_table(report: dict) -> None:
    """Print observation-type density (top 20)."""
    section = report.get("observation_density", {})
    obs_density = section.get("values", {})
    flagged_obs = set(section.get("flagged", {}).keys())

    tbl = Table(title="Observation-Type Density (top 20)")
    tbl.add_column("Obs Type", style="magenta")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Status")

    items = sorted(obs_density.items(), key=lambda x: -x[1])[:20]
    for obs_type, count in items:
        status = "[red]SPARSE[/]" if obs_type in flagged_obs else "[green]ok[/]"
        tbl.add_row(obs_type, str(count), status)
    console.print(tbl)


def _print_sparsity_table(report: dict) -> None:
    """Print neighborhood sparsity (mean degree per entity type)."""
    section = report.get("neighborhood_sparsity", {})
    sparsity = section.get("values", {})
    flagged_sparse = set(section.get("flagged", {}).keys())

    tbl = Table(title="Neighborhood Sparsity (mean degree per entity type)")
    tbl.add_column("Entity Type", style="green")
    tbl.add_column("Mean Degree", justify="right")
    tbl.add_column("Status")

    for etype, deg in sorted(sparsity.items(), key=lambda x: x[1]):
        status = "[red]SPARSE[/]" if etype in flagged_sparse else "[green]ok[/]"
        tbl.add_row(etype, f"{deg:.2f}", status)
    console.print(tbl)


def _print_recommendations(report: dict) -> None:
    """Print prioritised tool-wiring recommendations."""
    console.print("\n[bold cyan]═══ Recommendations ═══[/]")

    # Collect starved entity types — prefer those flagged for density AND sparsity
    density_sparse = set(report.get("entity_density", {}).get("flagged", {}).keys())
    degree_sparse = set(report.get("neighborhood_sparsity", {}).get("flagged", {}).keys())
    attn_low = set(report.get("edge_attention", {}).get("flagged", {}).keys())

    # Extract entity types from starved edges
    edge_entity_types: set[str] = set()
    for edge_str in attn_low:
        parts = edge_str.split("→")
        for p in parts:
            edge_entity_types.add(p.strip())

    priority_types = density_sparse | degree_sparse | edge_entity_types

    if not priority_types:
        console.print("[green]No starved entity types — graph health looks good.[/]")
        console.print("Next step: run Phase 49b (convergence control signal wiring).")
        return

    console.print(
        f"[yellow]Starved entity types: {', '.join(sorted(priority_types))}[/]\n"
    )
    for etype in sorted(priority_types):
        tools = _TOOL_SUGGESTIONS.get(etype, ["No specific tool suggestion"])
        console.print(f"[bold]{etype}[/]:")
        for t in tools:
            console.print(f"  • {t}")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GNN Attention Diagnostic — identify data-starved entity types"
    )
    parser.add_argument(
        "--db-path",
        default=".tirra_pipeline/pipeline.db",
        help="Path to PipelineStore SQLite DB",
    )
    parser.add_argument(
        "--model-path",
        default=".tirra_pipeline/gnn_model.pt",
        help="Path to trained GNN model checkpoint",
    )
    parser.add_argument(
        "--entity-density-min",
        type=int,
        default=5,
        help="Entity count below which a type is flagged SPARSE (default: 5)",
    )
    parser.add_argument(
        "--attention-min",
        type=float,
        default=_ATTENTION_LOW_THRESHOLD,
        help="Attention weight below which an edge type is flagged STARVED",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    model_path = Path(args.model_path)

    if not db_path.exists():
        console.print(f"[red]ERROR: DB not found at {db_path}[/]")
        sys.exit(1)

    console.print(f"\n[bold cyan]═══ GNN Attention Diagnostic ═══[/]")
    console.print(f"DB: {db_path}")
    console.print(f"Model: {model_path}\n")

    store = PipelineStore(str(db_path))
    try:
        # Load or build model
        if model_path.exists():
            console.print("[dim]Loading trained GNN model...[/]")
            try:
                trainer = Trainer.load_model(model_path, store)
                model = trainer.model
                console.print(
                    f"[green]Model loaded:[/] "
                    f"{sum(p.numel() for p in model.parameters()):,} params"
                )
            except Exception as e:
                console.print(f"[yellow]WARNING: Could not load model: {e}[/]")
                console.print(
                    "[dim]Building fresh model for graph structure analysis...[/]"
                )
                trainer = Trainer(store, TrainerConfig())
                trainer.build_model()
                model = trainer.model
        else:
            console.print(
                "[yellow]No trained model found — building fresh model for graph structure analysis.[/]"
            )
            trainer = Trainer(store, TrainerConfig())
            trainer.build_model()
            model = trainer.model

        # Run diagnostics
        console.print("[dim]Running diagnostics...[/]")
        diagnostics = compute_diagnostics(model, store)

        report = format_diagnostic_report(
            diagnostics,
            entity_density_min=args.entity_density_min,
            attention_min=args.attention_min,
        )

        # Print tables
        _print_entity_density_table(report)
        console.print()
        _print_obs_density_table(report)
        console.print()
        _print_attention_table(report)
        console.print()
        _print_sparsity_table(report)

        # Summary counts
        n_flagged_entity = len(report.get("entity_density", {}).get("flagged", {}))
        n_flagged_obs = len(report.get("observation_density", {}).get("flagged", {}))
        n_flagged_attn = len(report.get("edge_attention", {}).get("flagged", {}))
        n_flagged_degree = len(report.get("neighborhood_sparsity", {}).get("flagged", {}))

        console.print(
            f"\n[bold]Summary:[/] "
            f"{n_flagged_entity} sparse entity types, "
            f"{n_flagged_obs} sparse obs types, "
            f"{n_flagged_attn} starved edge types, "
            f"{n_flagged_degree} sparse neighborhoods"
        )

        _print_recommendations(report)

    finally:
        store.close()


if __name__ == "__main__":
    main()
