"""Tests for agent.quant.contract_opportunity — the Capital-pillar EV scorer."""

from __future__ import annotations

from agent.quant.contract_opportunity import (
    Opportunity,
    OpportunityRecord,
    opportunity_to_json,
    score_opportunities,
)


def test_expected_value_math():
    # EV = P(win) · (Bid − Cost) − Risk
    o = Opportunity(
        award_id="A1", recipient="R", agency="VA",
        description="", amount_usd=100000.0,
        start_date=None, award_type=None,
        bid_cost_usd=40000.0, risk_penalty=5000.0, own_win_rate=0.5,
    )
    assert o.expected_value == 0.5 * (100000 - 40000) - 5000


def test_estimated_cost_defaults_to_capped_amount():
    o = Opportunity(
        award_id="A1", recipient="R", agency="VA",
        description="", amount_usd=5000.0,
        start_date=None, award_type=None,
    )
    # bid_cost None → cost = min(amount, 10000) = 5000
    assert o.estimated_bid_cost_usd == 5000.0


def test_from_award_real_shape():
    o = Opportunity.from_award(
        {"award_id": "X1", "recipient": "Acme LLC", "agency": "USDA",
         "description": "janitorial", "amount_usd": 40000.0,
         "start_date": "2026-01-01", "award_type": None}
    )
    assert o.recipient == "Acme LLC"
    assert o.amount_usd == 40000.0
    assert o.award_type is None


def test_score_opportunities_ranks_by_ev():
    awards = [
        {"award_id": "big", "recipient": "A", "agency": "VA", "description": "x",
         "amount_usd": 200000.0, "start_date": None, "award_type": None},
        {"award_id": "small", "recipient": "B", "agency": "USDA", "description": "y",
         "amount_usd": 30000.0, "start_date": None, "award_type": None},
    ]
    ranked = score_opportunities(awards)
    assert [o.award_id for o in ranked] == ["big", "small"]
    assert ranked[0].expected_value > ranked[1].expected_value


def test_to_json():
    o = Opportunity.from_award({"award_id": "X", "recipient": "R", "agency": "A",
                                "description": "d", "amount_usd": 1000.0,
                                "start_date": None, "award_type": None})
    row = opportunity_to_json([o])[0]
    assert "expected_value_usd" in row
    assert "p_win" in row


def test_record_roundtrip_dict():
    r = OpportunityRecord(award_id="X", expected_value_usd=123.0)
    assert r.to_dict()["award_id"] == "X"


from agent.quant.contract_opportunity import (
    WinProbabilityLearner,
    apply_learned_probabilities,
)


def test_learner_cold_start_prior():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    learner = WinProbabilityLearner(os.path.join(d, "w.jsonl"))
    # Prior Beta(0.5, 1.0) → mean 1/3, NOT the naive 0.5 coin-flip prior.
    assert abs(learner.probability_of("VA", 50000.0) - 1.0 / 3.0) < 1e-9
    # basis says prior, not learned
    assert learner.basis_of("VA", 50000.0)["source"] == "prior"


def test_learner_updates_with_evidence():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    learner = WinProbabilityLearner(os.path.join(d, "w.jsonl"))
    for i in range(10):
        learner.record(f"a{i}", "VA", 50000.0, realized_success=(i < 8))
    p = learner.probability_of("VA", 50000.0)
    assert p > 0.5  # 8 wins / 2 losses → posterior mean above prior
    # posterior = (0.5 + 8) / (1.0 + 10) = 8.5/11.5 ≈ 0.7391
    assert abs(p - 0.7391304348) < 1e-6
    # basis now learned, with counts
    basis = learner.basis_of("VA", 50000.0)
    assert basis["source"] == "learned"
    assert basis["wins"] == 8 and basis["bids"] == 10
    # different agency untouched → still prior mean
    assert abs(learner.probability_of("USDA", 50000.0) - 1.0 / 3.0) < 1e-9


def test_learner_persists():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "w.jsonl")
    learner = WinProbabilityLearner(path)
    for i in range(6):
        learner.record(f"a{i}", "DOD", 200000.0, realized_success=(i < 5))
    p1 = learner.probability_of("DOD", 200000.0)
    reloaded = WinProbabilityLearner(path)
    assert abs(reloaded.probability_of("DOD", 200000.0) - p1) < 1e-9


def test_apply_learned_probabilities_reranks():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    learner = WinProbabilityLearner(os.path.join(d, "w.jsonl"))
    # learner: VA wins often, USDA loses
    for i in range(6):
        learner.record(f"a{i}", "VA", 50000.0, realized_success=(i < 5))
        learner.record(f"b{i}", "USDA", 50000.0, realized_success=False)
    awards = [
        {"award_id": "VA1", "recipient": "R1", "agency": "VA", "description": "d",
         "amount_usd": 50000.0, "start_date": None, "award_type": None},
        {"award_id": "US1", "recipient": "R2", "agency": "USDA", "description": "d",
         "amount_usd": 50000.0, "start_date": None, "award_type": None},
    ]
    opps = apply_learned_probabilities(score_opportunities(awards), learner)
    ranked_ids = [o.award_id for o in opps]
    assert ranked_ids[0] == "VA1"  # higher learned P(win) → higher EV
