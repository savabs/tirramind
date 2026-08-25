"""
M15 quant data universe — must-add vs should-add tickers (full-quant base model).

Verified 2026-06-03 via yfinance probe:
  - Options chains: SPY, QQQ, IWM, GLD, SLV, ^SPX, ^VIX, CL (not CL=F futures symbol)
  - Futures =F symbols generally have 0 option expiries on Yahoo
"""

from __future__ import annotations

from agent.tools.instrument_universe import (
    INSTRUMENTS,
    InstrumentDef,
    tradeable_instruments,
)

# Must-add (T0) — options + vol indices for base quant model
OPTIONS_MUST: tuple[str, ...] = ("SPY", "^VIX")

# Should-add (T1) — broaden equity/commodity vol surface
OPTIONS_SHOULD: tuple[str, ...] = ("QQQ", "IWM", "GLD", "SLV", "^SPX", "CL")

OPTIONS_DEFAULT: tuple[str, ...] = OPTIONS_MUST + OPTIONS_SHOULD

# Dividends — equity ETFs + flagship index ETF (must SPY; should rest of US equity_etf)
DIVIDEND_MUST: tuple[str, ...] = ("SPY",)
DIVIDEND_SHOULD: tuple[str, ...] = tuple(
    sorted(
        {
            i.ticker
            for i in INSTRUMENTS
            if i.asset_class == "equity_etf" and i.ticker != "SPY"
        }
    )
)

DIVIDEND_DEFAULT: tuple[str, ...] = DIVIDEND_MUST + DIVIDEND_SHOULD


def all_options_tickers(*, include_should: bool = True) -> list[str]:
    if include_should:
        return list(OPTIONS_DEFAULT)
    return list(OPTIONS_MUST)


def all_dividend_tickers(*, include_should: bool = True) -> list[str]:
    if include_should:
        return list(DIVIDEND_DEFAULT)
    return list(DIVIDEND_MUST)


def instrument_def_for_ticker(ticker: str) -> InstrumentDef | None:
    for inst in INSTRUMENTS:
        if inst.ticker == ticker:
            return inst
    return None


def tradeable_ticker_set() -> set[str]:
    return {i.ticker for i in tradeable_instruments()}
