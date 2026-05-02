"""
TirraMind Agent — CLI Entry Point

Usage:
    python -m agent.cli "Your goal here"
    python -m agent.cli --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from agent.config.settings import AgentConfig
from agent.core.autonomous import AutonomousRunner, LoopIteration
from agent.core.orchestrator import Orchestrator
from agent.data.cache import DataCache
from agent.pipeline.executor import DAGExecutor
from agent.pipeline.store import PipelineStore
from agent.tools.academic_preprints import AcademicPreprintsTool
from agent.tools.ais_vessel import AISVesselTool
from agent.tools.backtest import BacktestTool
from agent.tools.bankruptcy_court import BankruptcyCourtTool
from agent.tools.base import ToolRegistry
from agent.tools.building_permits import BuildingPermitsTool
from agent.tools.capital_flows import CapitalFlowsTool
from agent.tools.central_bank_balance import CentralBankBalanceTool
from agent.tools.cert_transparency import CertTransparencyTool
from agent.tools.cftc import CFTCTool
from agent.tools.code_executor import CodeExecutorTool
from agent.tools.comtrade import ComtradeTool
from agent.tools.consumer_sentiment import ConsumerSentimentTool
from agent.tools.creditor_filings import CreditorFilingsTool
from agent.tools.defi_flows import DefiFlowsTool
from agent.tools.disease_surveillance import DiseaseSurveillanceTool
from agent.tools.dns_monitor import DnsMonitorTool
from agent.tools.drug_regulatory import DrugRegulatoryTool
from agent.tools.earthquake_proximity import EarthquakeProximityTool
from agent.tools.electricity_monitor import ElectricityMonitorTool
from agent.tools.energy_supply import EnergySupplyTool
from agent.tools.file_manager import FileReadTool, FileWriteTool, ListDirectoryTool
from agent.tools.finra_short_volume import FinraShortVolumeTool
from agent.tools.foia_requests import FoiaRequestsTool
from agent.tools.food_security import FoodSecurityTool
from agent.tools.form144 import Form144Tool
from agent.tools.gdelt import GDELTTool
from agent.tools.global_pmi import GlobalPmiTool
from agent.tools.gov_contracts import GovContractsTool
from agent.tools.insider_filings import InsiderFilingsTool
from agent.tools.interconnection_queue import InterconnectionQueueTool
from agent.tools.internet_infrastructure import InternetInfrastructureTool
from agent.tools.internet_outages import InternetOutagesTool
from agent.tools.job_postings import JobPostingsTool
from agent.tools.labor_disruptions import LaborDisruptionsTool
from agent.tools.liquidity_regime import LiquidityRegimeTool
from agent.tools.lobbying import LobbyingTool
from agent.tools.macro_data import MacroDataTool
from agent.tools.market_data import MarketDataTool
from agent.tools.migration_flows import MigrationFlowsTool
from agent.tools.patent_filings import PatentFilingsTool
from agent.tools.pipeline_query import PipelineQueryTool
from agent.tools.political_risk import PoliticalRiskTool
from agent.tools.polymarket import PolymarketTool
from agent.tools.polymarket_whales import PolymarketWhalesTool
from agent.tools.power_grid import PowerGridTool
from agent.tools.regulatory_gazette import RegulatoryGazetteTool
from agent.tools.sanctions_monitor import SanctionsMonitorTool
from agent.tools.satellite_activity import SatelliteActivityTool
from agent.tools.shell_runner import ShellRunnerTool
from agent.tools.sovereign_debt import SovereignDebtTool
from agent.tools.supply_chain_monitor import SupplyChainMonitorTool
from agent.tools.transport_throughput import TransportThroughputTool
from agent.tools.treasury_receipts import TreasuryReceiptsTool
from agent.tools.weather_alerts import WeatherAlertsTool
from agent.tools.web_browse import WebBrowseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.whale_alert import WhaleAlertTool
from agent.tools.wikipedia_pageviews import WikipediaPageviewsTool

console = Console()


def build_tool_registry(config: AgentConfig | None = None) -> ToolRegistry:
    """Register all available tools."""
    cache = DataCache()
    registry = ToolRegistry()

    # ── PipelineStore: create early so tools can persist entities ──
    pipeline_store = PipelineStore(db_path=config.pipeline.db_path if config else ".tirra_pipeline/pipeline.db")

    registry.register(WebSearchTool())
    registry.register(WebBrowseTool())
    timeout = config.tool_timeout if config else 30
    registry.register(CodeExecutorTool(timeout=timeout))
    registry.register(ShellRunnerTool(timeout=timeout))
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(ListDirectoryTool())
    registry.register(MarketDataTool(cache=cache))
    fred_key = config.fred_api_key if config else ""
    registry.register(MacroDataTool(fred_api_key=fred_key, cache=cache))
    registry.register(LiquidityRegimeTool(fred_api_key=fred_key, cache=cache))
    registry.register(BacktestTool(fred_api_key=fred_key, cache=cache))
    registry.register(PolymarketTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(
        PolymarketWhalesTool(
            db_path=(config.pipeline.db_path if config else ".tirra_pipeline/pipeline.db"),
        )
    )
    registry.register(InsiderFilingsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(GDELTTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(CFTCTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(WhaleAlertTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(Form144Tool(cache=cache, pipeline_store=pipeline_store))
    registry.register(FinraShortVolumeTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(PowerGridTool(cache=cache))
    registry.register(WikipediaPageviewsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(AISVesselTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(RegulatoryGazetteTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(WeatherAlertsTool(cache=cache))
    registry.register(EarthquakeProximityTool(cache=cache))
    registry.register(TransportThroughputTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(DefiFlowsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(GovContractsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(AcademicPreprintsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(SanctionsMonitorTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(CertTransparencyTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(BankruptcyCourtTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(DnsMonitorTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(SovereignDebtTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(CentralBankBalanceTool(fred_api_key=fred_key, cache=cache, pipeline_store=pipeline_store))
    registry.register(FoiaRequestsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(CreditorFilingsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(ComtradeTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(JobPostingsTool(fred_api_key=fred_key, cache=cache))
    registry.register(BuildingPermitsTool(fred_api_key=fred_key, cache=cache))
    registry.register(CapitalFlowsTool(fred_api_key=fred_key, cache=cache, pipeline_store=pipeline_store))
    registry.register(PatentFilingsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(LobbyingTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(SatelliteActivityTool(cache=cache))
    registry.register(ElectricityMonitorTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(InterconnectionQueueTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(DiseaseSurveillanceTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(FoodSecurityTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(PoliticalRiskTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(InternetOutagesTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(LaborDisruptionsTool(cache=cache))
    registry.register(MigrationFlowsTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(EnergySupplyTool(cache=cache))
    registry.register(TreasuryReceiptsTool(cache=cache))
    registry.register(DrugRegulatoryTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(GlobalPmiTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(ConsumerSentimentTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(SupplyChainMonitorTool(cache=cache, pipeline_store=pipeline_store))
    registry.register(InternetInfrastructureTool(cache=cache))
    registry.register(PipelineQueryTool(store=pipeline_store))

    # ── Tier 8, Change 16: Seed OntologyRegistry on startup ───
    try:
        from agent.discovery.ontology_registry import OntologyRegistry
        from agent.pipeline.entity import set_ontology_registry

        ontology = OntologyRegistry(pipeline_store)
        set_ontology_registry(ontology)
    except Exception:
        logging.getLogger(__name__).debug("OntologyRegistry init skipped (discovery module unavailable)")

    # ── Tier 8, Change 15: Load discovered tool configs ───────
    try:
        from agent.discovery.tool_factory import ToolFactory

        factory = ToolFactory()
        for tool in factory.load_all_configs():
            # Only register tools whose source is active in the store
            sources = pipeline_store.query_discovered_sources(status="active")
            active_ids = {s["source_id"] for s in sources}
            # Tool names are "discovered_{source_id[:8]}"
            source_prefix = tool.name.replace("discovered_", "")
            if any(sid.startswith(source_prefix) for sid in active_ids):
                registry.register(tool)
    except Exception:
        logging.getLogger(__name__).debug("Discovered tool loading skipped (no configs or module unavailable)")

    return registry


def on_step(step: int, task, result) -> None:
    """Real-time progress callback."""
    status = "[green]✓[/]" if result.success else "[red]✗[/]"
    console.print(f"  {status} Step {step}: [bold]{task.tool}[/] → {task.description[:60]}")
    if not result.success:
        console.print(f"    [red]{result.output[:120]}[/]")


def run_goal(goal: str, config: AgentConfig) -> None:
    """Execute a single goal."""
    console.print(Panel(f"[bold cyan]Goal:[/] {goal}", title="TirraMind Agent", border_style="cyan"))

    registry = build_tool_registry(config)
    console.print(f"[dim]Tools loaded: {', '.join(registry.list_names())}[/]")
    console.print(f"[dim]LLM: {config.llm.provider}/{config.llm.model}[/]")
    console.print()

    agent = Orchestrator(config=config, tool_registry=registry)
    result = agent.run(goal, on_step=on_step)

    console.print()
    if result.success:
        console.print(Panel(result.output, title="[green]Result[/]", border_style="green"))
    else:
        console.print(Panel(result.output, title="[yellow]Partial Result[/]", border_style="yellow"))

    console.print(f"\n[dim]Steps: {result.steps_taken} | Success: {result.success}[/]")


def run_autonomous(config: AgentConfig, max_goals: int) -> None:
    """Run the agent in autonomous mode — self-directed goal loop."""
    console.print(
        Panel(
            f"[bold cyan]Autonomous Mode[/] — max goals: {max_goals}",
            title="TirraMind Agent",
            border_style="magenta",
        )
    )

    registry = build_tool_registry(config)
    console.print(f"[dim]Tools loaded: {', '.join(registry.list_names())}[/]")
    console.print(f"[dim]LLM: {config.llm.provider}/{config.llm.model}[/]")
    console.print()

    def on_iteration(it: LoopIteration) -> None:
        status = "[green]✓[/]" if it.evaluation.success else "[red]✗[/]"
        console.print(
            f"\n  {status} Iteration {it.iteration} "
            f"[magenta]\\[{it.arm.name}][/]: "
            f"[bold]{it.goal.description[:60]}[/] "
            f"(score={it.evaluation.score:.2f}, reward={it.reward:.3f})"
        )
        if it.evaluation.lessons:
            for lesson in it.evaluation.lessons[:2]:
                console.print(f"    [dim]→ {lesson}[/]")

    runner = AutonomousRunner(
        config=config,
        tool_registry=registry,
        max_iterations=max_goals,
        on_iteration=on_iteration,
    )
    summary = runner.run()

    console.print()
    console.print(
        Panel(
            summary.report(),
            title="[magenta]Autonomous Run Complete[/]",
            border_style="magenta",
        )
    )


def run_interactive(config: AgentConfig) -> None:
    """Interactive REPL mode."""
    console.print(
        Panel(
            "[bold cyan]TirraMind Agent — Interactive Mode[/]\nType a goal and press Enter. Type 'quit' to exit.",
            border_style="cyan",
        )
    )

    while True:
        try:
            goal = console.input("\n[bold green]Goal >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not goal or goal.lower() in ("quit", "exit", "q"):
            break

        run_goal(goal, config)


def run_pipeline(args_pipeline: list[str], config: AgentConfig) -> None:
    """Handle --pipeline sub-commands: run, list, status, start."""
    if not args_pipeline:
        console.print("[red]Usage: --pipeline <run|list|status|start>[/]")
        sys.exit(1)

    sub = args_pipeline[0]

    if sub == "run":
        _pipeline_run(args_pipeline[1:], config)
    elif sub == "list":
        _pipeline_list()
    elif sub == "status":
        _pipeline_status(args_pipeline[1:], config)
    elif sub == "start":
        _pipeline_start(config)
    else:
        console.print(f"[red]Unknown pipeline command: {sub}[/]")
        console.print("[dim]Available: run, list, status, start[/]")
        sys.exit(1)


def _pipeline_run(rest: list[str], config: AgentConfig) -> None:
    """Execute a DAG manually by name."""
    if not rest:
        console.print("[red]Usage: --pipeline run <dag_name>[/]")
        sys.exit(1)

    dag_name = rest[0]
    registry = build_tool_registry(config)

    # Lazy import to avoid circular deps / missing registry module
    try:
        from agent.pipeline.registry import DAGRegistry
    except ImportError:
        console.print("[red]DAGRegistry not yet implemented (Step 7.7).[/]")
        sys.exit(1)

    dag_registry = DAGRegistry()
    dag_registry.load_defaults(registry)
    dag = dag_registry.get(dag_name)
    if dag is None:
        console.print(f"[red]DAG not found: {dag_name}[/]")
        console.print(f"[dim]Available: {', '.join(d.name for d in dag_registry.list_all())}[/]")
        sys.exit(1)

    store = PipelineStore(db_path=config.pipeline.db_path)
    executor = DAGExecutor(
        tool_registry=registry,
        store=store,
        max_workers=config.pipeline.max_workers,
    )

    console.print(
        Panel(
            f"[bold cyan]Pipeline Run:[/] {dag_name}",
            title="TirraMind Pipeline",
            border_style="blue",
        )
    )
    result = executor.execute(dag, trigger="manual")
    status_color = "green" if result.status == "success" else "red"
    console.print(f"\n[{status_color}]Run {result.run_id}: {result.status}[/]")
    for nid, nr in result.node_results.items():
        icon = "[green]✓[/]" if nr.status == "success" else "[red]✗[/]"
        console.print(f"  {icon} {nid}: {nr.status}")


def _pipeline_list() -> None:
    """Show registered DAGs."""
    try:
        from agent.pipeline.registry import DAGRegistry
    except ImportError:
        console.print("[red]DAGRegistry not yet implemented (Step 7.7).[/]")
        sys.exit(1)

    dag_registry = DAGRegistry()
    # Load defaults without tools — we just need DAG metadata
    try:
        dag_registry.load_defaults(ToolRegistry())
    except Exception:
        pass

    dags = dag_registry.list_all()
    if not dags:
        console.print("[dim]No DAGs registered.[/]")
        return

    console.print("[bold]Registered DAGs:[/]")
    for d in dags:
        console.print(
            f"  [cyan]{d.name}[/] — {d.description or '(no description)'} [dim]schedule={d.schedule or 'none'}[/]"
        )


def _pipeline_status(rest: list[str], config: AgentConfig) -> None:
    """Show pipeline run history or a specific run."""
    store = PipelineStore(db_path=config.pipeline.db_path)

    if rest:
        run_id = rest[0]
        run = store.get_run(run_id)
        if run is None:
            console.print(f"[red]Run not found: {run_id}[/]")
            sys.exit(1)
        console.print(
            Panel(
                f"[bold]Run:[/] {run['run_id']}\n"
                f"[bold]DAG:[/] {run['dag_name']}\n"
                f"[bold]Status:[/] {run['status']}\n"
                f"[bold]Trigger:[/] {run['trigger']}\n"
                f"[bold]Started:[/] {run['started_at']}\n"
                f"[bold]Ended:[/] {run.get('ended_at', 'in-progress')}",
                title="Pipeline Run",
                border_style="blue",
            )
        )
    else:
        runs = store.get_runs(limit=20)
        if not runs:
            console.print("[dim]No pipeline runs found.[/]")
            return
        console.print("[bold]Recent Pipeline Runs:[/]")
        for r in runs:
            status_color = "green" if r["status"] == "success" else "yellow" if r["status"] == "running" else "red"
            console.print(
                f"  [{status_color}]{r['status']:<8}[/] "
                f"[cyan]{r['dag_name']}[/] "
                f"[dim]{r['run_id'][:12]}… {r['started_at']}[/]"
            )


def _pipeline_start(config: AgentConfig) -> None:
    """Start the pipeline scheduler (blocks until Ctrl+C)."""
    try:
        from agent.pipeline.registry import DAGRegistry
        from agent.pipeline.scheduler import PipelineScheduler
    except ImportError:
        console.print("[red]PipelineScheduler or DAGRegistry not yet implemented.[/]")
        sys.exit(1)

    registry = build_tool_registry(config)
    store = PipelineStore(db_path=config.pipeline.db_path)
    executor = DAGExecutor(
        tool_registry=registry,
        store=store,
        max_workers=config.pipeline.max_workers,
    )
    dag_registry = DAGRegistry()
    dag_registry.load_defaults(registry)

    scheduler = PipelineScheduler(executor=executor, registry=dag_registry)
    console.print(
        Panel(
            "[bold cyan]Pipeline Scheduler — Starting[/]\nPress Ctrl+C to stop.",
            title="TirraMind Pipeline",
            border_style="blue",
        )
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.stop()
        console.print("\n[dim]Scheduler stopped.[/]")


def main() -> None:
    # Load .env before parsing config
    load_dotenv(Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="TirraMind Autonomous Intelligence Agent")
    parser.add_argument("goal", nargs="?", help="The goal for the agent to accomplish")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--autonomous", "-a", action="store_true", help="Autonomous self-directed mode")
    parser.add_argument(
        "--max-goals",
        type=int,
        default=5,
        help="Max goals in autonomous mode (default: 5)",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        nargs="*",
        metavar="CMD",
        help="Pipeline mode: run <dag>, list, status [run_id], start",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = AgentConfig.from_env()

    # Validate config before running
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"[red bold]Config error:[/] {err}")
        console.print("\n[dim]See .env.example for required settings.[/]")
        sys.exit(1)

    if args.pipeline is not None:
        run_pipeline(args.pipeline, config)
    elif args.autonomous:
        run_autonomous(config, max_goals=args.max_goals)
    elif args.interactive:
        run_interactive(config)
    elif args.goal:
        run_goal(args.goal, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
