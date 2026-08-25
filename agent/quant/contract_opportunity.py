"""ContractOpportunity — expected-value scoring over real government contract awards.

The Capital pillar starter kit: turns raw `gov_contracts` award observations into
an expected-value opportunity score, wired back through the (now-proven) learning loop so
that every scored opportunity is also an input that makes the loop compound.

   EV = P(win) · (Bid − Cost) − Risk

All parameters are explicit, deterministic, and inspectable — no hidden LLM opinion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_WIN_RATE = 0.3
_AMOUNT_BUCKETS = (25_000.0, 100_000.0, 500_000.0, float("inf"))


def _amount_bucket(amount: float | None) -> str:
    a = amount or 0.0
    for b in _AMOUNT_BUCKETS:
        if a <= b:
            return f"le_{b:g}"
    return "gt_500000"


@dataclass(frozen=True)
class Opportunity:
    """One scored government-contract opportunity."""

    award_id: str
    recipient: str
    agency: str
    description: str
    amount_usd: float | None
    start_date: str | None
    award_type: str | None

    # Scoring inputs (all explicit, defaulted conservatively)
    bid_cost_usd: float | None = field(default=None)
    risk_penalty: float = 0.0
    own_win_rate: float = _DEFAULT_WIN_RATE
    p_win_basis: str = "default"  # "default" | "prior" | "learned"

    @classmethod
    def from_award(cls, award: dict[str, Any]) -> Opportunity:
        return cls(
            award_id=str(award.get("award_id") or ""),
            recipient=str(award.get("recipient") or ""),
            agency=str(award.get("agency") or ""),
            description=str(award.get("description") or ""),
            amount_usd=_to_float(award.get("amount_usd")),
            start_date=str(award.get("start_date") or "") or None,
            award_type=str(award.get("award_type") or "") or None,
        )

    @property
    def estimated_bid_cost_usd(self) -> float:
        if self.bid_cost_usd is not None:
            return self.bid_cost_usd
        base = self.amount_usd or 5000.0
        return min(base, 10000.0)

    @property
    def expected_value(self) -> float:
        """EV = P(win) · (Bid − Cost) − Risk (deterministic; no LLM)."""
        p_win = min(1.0, max(0.0, self.own_win_rate))
        gross_profit = (self.amount_usd or 0.0) - self.estimated_bid_cost_usd
        return float(p_win * gross_profit - self.risk_penalty)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["description"] = (self.description or "")[:160]
        d["estimated_bid_cost_usd"] = round(self.estimated_bid_cost_usd, 2)
        d["bucket"] = _amount_bucket(self.amount_usd)
        d["p_win"] = self.own_win_rate
        d["expected_value_usd"] = round(self.expected_value, 2)
        return d

    def to_ranked(self) -> tuple[str, float, dict[str, Any]]:
        return (self.award_id, self.expected_value, self.as_dict())


class WinProbabilityLearner:
    """Learned P(win) per (agency, amount-bucket) from realized outcomes,
    with a documented prior instead of a naive flat 0.5.

    Prior: ``prior_wins`` and ``prior_bids`` define a Beta(prior_wins,
    prior_bids) starting belief (posterior mean = prior_wins/prior_bids, e.g.
    0.5/1.0 → 0.33). With zero realized outcomes every agency+bucket shows this
    prior (lower than the naive 0.5, reflecting that a random small-business
    bid is not an even coin flip). Each realized outcome shifts the posterior:
    success increments wins, failure increments bids.

    ``basis`` is exposed so consumers know whether a P(win) is evidence-backed
    ("learned:N") or still a prior ("prior") — honest, no hidden confidence.
    """

    def __init__(
        self,
        store_path: str = ".tirra_opportunities/win_learner.jsonl",
        prior_wins: float = 0.5,
        prior_bids: float = 1.0,
    ) -> None:
        if prior_bids <= 0 or prior_wins < 0:
            raise ValueError("prior_bids must be > 0 and prior_wins >= 0")
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prior_wins = float(prior_wins)
        self._prior_bids = float(prior_bids)
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        # Track realized counts per key (wins, total) for an honest basis.
        self._evidence: dict[str, tuple[int, int]] = {}
        self._outcomes: list[dict[str, Any]] = []
        self._load()

    @staticmethod
    def _key(agency: str, amount: float | None) -> str:
        return f"{agency or '?'}::{_amount_bucket(amount)}"

    def record(self, award_id: str, agency: str, amount: float | None, realized_success: bool) -> None:
        k = self._key(agency, amount)
        a = self._alpha.get(k, self._prior_wins)
        b = self._beta.get(k, self._prior_bids)
        ev = self._evidence.get(k, (0, 0))
        if realized_success:
            a += 1.0
            ev = (ev[0] + 1, ev[1] + 1)
        else:
            b += 1.0
            ev = (ev[0], ev[1] + 1)
        self._alpha[k] = a
        self._beta[k] = b
        self._evidence[k] = ev
        rec = {
            "award_id": award_id,
            "agency": agency or "?",
            "amount": amount,
            "amount_bucket": _amount_bucket(amount),
            "realized_success": bool(realized_success),
        }
        self._outcomes.append(rec)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def probability_of(self, agency: str, amount: float | None) -> float:
        """Posterior-mean P(win) for this agency+bucket (Beta posterior)."""
        k = self._key(agency, amount)
        a = self._alpha.get(k, self._prior_wins)
        b = self._beta.get(k, self._prior_bids)
        return a / (a + b)

    def basis_of(self, agency: str, amount: float | None) -> dict[str, Any]:
        """Expose whether P(win) is prior-backed or learned, with evidence count."""
        k = self._key(agency, amount)
        wins, total = self._evidence.get(k, (0, 0))
        if total == 0:
            return {
                "source": "prior",
                "prior_mean": self._prior_wins / self._prior_bids,
                "learned_wins": 0,
                "learned_bids": 0,
            }
        return {
            "source": "learned",
            "wins": wins,
            "bids": total,
            "prior_mean": self._prior_wins / self._prior_bids,
        }

    def state(self) -> dict[str, Any]:
        return {
            "prior_wins": self._prior_wins,
            "prior_bids": self._prior_bids,
            "alpha": dict(self._alpha),
            "beta": dict(self._beta),
            "n_outcomes": len(self._outcomes),
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        alpha: dict[str, float] = {}
        beta: dict[str, float] = {}
        evidence: dict[str, tuple[int, int]] = {}
        outcomes: list[dict[str, Any]] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                outcomes.append(rec)
                k = self._key(str(rec.get("agency", "?")), rec.get("amount"))
                a = alpha.get(k, self._prior_wins)
                b = beta.get(k, self._prior_bids)
                ev = evidence.get(k, (0, 0))
                if rec.get("realized_success"):
                    a += 1.0
                    ev = (ev[0] + 1, ev[1] + 1)
                else:
                    b += 1.0
                    ev = (ev[0], ev[1] + 1)
                alpha[k] = a
                beta[k] = b
                evidence[k] = ev
        except Exception:
            pass
        self._alpha, self._beta, self._evidence, self._outcomes = alpha, beta, evidence, outcomes


def apply_learned_probabilities(
    opps: list[Opportunity],
    learner: WinProbabilityLearner,
) -> list[Opportunity]:
    """Return new Opportunities with P(win) from the learned learner."""
    rebuilt: list[Opportunity] = []
    for o in opps:
        p_win = learner.probability_of(o.agency, o.amount_usd)
        basis = learner.basis_of(o.agency, o.amount_usd)
        rebuilt.append(
            Opportunity(
                award_id=o.award_id,
                recipient=o.recipient,
                agency=o.agency,
                description=o.description,
                amount_usd=o.amount_usd,
                start_date=o.start_date,
                award_type=o.award_type,
                bid_cost_usd=o.bid_cost_usd,
                risk_penalty=o.risk_penalty,
                own_win_rate=p_win,
                p_win_basis=basis["source"],
            )
        )
    rebuilt.sort(key=lambda x: x.expected_value, reverse=True)
    return rebuilt


def score_opportunities(awards: list[dict[str, Any]]) -> list[Opportunity]:
    """Scored opportunities, sorted by EV descending (default P(win)=0.3)."""
    opps = [Opportunity.from_award(a) for a in awards]
    return sorted(opps, key=lambda o: o.expected_value, reverse=True)


def opportunity_to_json(opps: list[Opportunity]) -> list[dict[str, Any]]:
    return [o.as_dict() for o in opps]


@dataclass(frozen=True)
class OpportunityRecord:
    """Persisted outcome record fed back into the learning loop."""

    award_id: str
    expected_value_usd: float
    agency: str | None = None
    amount_usd: float | None = None
    realized_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "Opportunity",
    "OpportunityRecord",
    "WinProbabilityLearner",
    "apply_learned_probabilities",
    "score_opportunities",
    "opportunity_to_json",
]
