"""Long-running daemon.

Uses APScheduler's BlockingScheduler for periodic watcher ticks and
dispatcher drains. Each tick is short; if a watcher is busy its next
tick is simply skipped (coalesce=True, max_instances=1).
"""

from __future__ import annotations

import logging
import signal
from datetime import UTC
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

from agent.awos.classifiers.composite import CompositeClassifier
from agent.awos.config import AWOSConfig
from agent.awos.events.bus import EventBus
from agent.awos.orchestrator.dispatcher import Dispatcher
from agent.awos.policies.engine import PolicyEngine
from agent.awos.watchers.chat_log import ChatLogWatcher
from agent.awos.watchers.drift import DriftWatcher
from agent.awos.watchers.obsidian import ObsidianWatcher
from agent.awos.watchers.staleness import StalenessWatcher

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, cfg: AWOSConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.bus = EventBus(
            cfg.db_path,  # type: ignore[arg-type]
            dedup_window_s=cfg.dedup_window_s,
            max_payload_bytes=cfg.max_payload_bytes,
        )
        policies_file = cfg.policies_file
        self.policies = PolicyEngine.load(policies_file)
        self.dispatcher = Dispatcher(cfg, self.bus, self.policies)
        self.classifier = CompositeClassifier.from_config(cfg)

        self.watchers = []
        if cfg.drift_watcher_enabled:
            self.watchers.append(DriftWatcher(self.bus, cfg.repo_root, timeout=cfg.watcher_timeout_s))
        if cfg.obsidian_watcher_enabled:
            self.watchers.append(ObsidianWatcher(self.bus, cfg.repo_root, timeout=cfg.watcher_timeout_s))
        if cfg.staleness_watcher_enabled:
            self.watchers.append(
                StalenessWatcher(
                    self.bus,
                    cfg.repo_root,
                    stale_task_days=cfg.stale_task_days,
                )
            )
        if cfg.chat_log_watcher_enabled:
            self.watchers.append(
                ChatLogWatcher(
                    self.bus,
                    cfg.repo_root,
                    classifier=self.classifier,
                    state_file=cfg.state_file,  # type: ignore[arg-type]
                    log_dir=cfg.chat_log_dir,
                )
            )

        self.scheduler = BlockingScheduler(timezone="UTC")

    # ------------------------------------------------------------------
    def _tick_watchers(self) -> None:
        for w in self.watchers:
            n = w.run_once()
            if n:
                log.info("watcher %s produced %d events", w.name, n)

    def _tick_dispatch(self) -> None:
        reports = self.dispatcher.drain(limit=50)
        if reports:
            log.info("dispatched %d events", len(reports))

    # ------------------------------------------------------------------
    def start(self) -> None:
        interval = self.cfg.watcher_interval_s
        self.scheduler.add_job(
            self._tick_watchers,
            "interval",
            seconds=interval,
            id="watchers",
            coalesce=True,
            max_instances=1,
            next_run_time=_now_utc(),
        )
        self.scheduler.add_job(
            self._tick_dispatch,
            "interval",
            seconds=max(15, interval // 4),
            id="dispatch",
            coalesce=True,
            max_instances=1,
            next_run_time=_now_utc(),
        )
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self.bus.close()


def _now_utc():
    from datetime import datetime

    return datetime.now(UTC)


# convenience for tests
def build(cfg: AWOSConfig) -> Daemon:
    return Daemon(cfg)


def run_once(cfg: AWOSConfig) -> tuple[int, int]:
    """Non-blocking single-pass: scan + dispatch. Returns (events, dispatched)."""
    d = Daemon(cfg)
    events = 0
    for w in d.watchers:
        events += w.run_once()
    reports = d.dispatcher.drain(limit=100)
    d.bus.close()
    return events, len(reports)


def _install_path(path: Path) -> Path:  # pragma: no cover - unused
    return path


__all__ = ["Daemon", "build", "run_once"]
