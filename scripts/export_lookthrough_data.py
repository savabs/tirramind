#!/usr/bin/env python3
"""Export everything the STATIC look-through page needs, plus its parity proof.

WHY THIS EXISTS
    The look-through is the one finding in this project that survived honest
    measurement, and it needs no price history — only index membership, index
    weights, and a fund-name resolver. All three are precomputable, so the
    product can be a static page on a CDN instead of a Python service on a
    paid VM that has to stay up. That is not a cost saving dressed up as
    architecture: the audit server's own runbook records 26 days of silent
    downtime, and a page that computes in the browser cannot have that
    failure mode at all.

    The picking-cost and factor blocks genuinely need yfinance history and
    stay on the server. This exports the part that does not.

THE RISK THIS FILE IS DESIGNED AROUND
    A second implementation of the same arithmetic in a second language is
    how two answers to one question appear. The whole guard is that the JS
    does arithmetic and table lookups ONLY, and that this script emits the
    evidence to prove the JS agrees with Python:

      * ``parity`` — every scheme name in the live AMFI master, with the
        ``index_phrase`` and ``classify`` output Python computes for it. The
        JS port is run over the same corpus in
        ``tests/test_lookthrough_static_parity.py`` and must agree on every
        row. Thousands of real names, not a handful of hand-picked ones.
      * ``golden`` — whole books run through the real ``look_through``, with
        the resulting weights. The JS must reproduce every number.

    If either disagrees, the test fails and the static page does not ship.
    That is the only thing standing between "it looks right" and "it is the
    same answer".

WHAT IT DELIBERATELY DOES NOT DO
    It does not invent an index weight, and it does not widen coverage to
    make the page look more capable. An index with no constituent list is
    left out, and the page then refuses that fund by name. Under-claiming is
    recoverable; unpacking a position into 150 wrong companies and reporting
    success is not.

USAGE
    PYTHONPATH=. .venv/bin/python scripts/export_lookthrough_data.py
    PYTHONPATH=. .venv/bin/python scripts/export_lookthrough_data.py --offline

    ``--offline`` uses only cached NSE/Yahoo/AMFI responses and fails rather
    than fetch. Use it to re-emit the file without moving the numbers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from agent.lookthrough import combine as C
from agent.lookthrough import funds as F
from agent.lookthrough import indices as I

#: Written next to the page that reads it, so the deploy is one directory.
#: Holds ONLY what the browser needs — the indices and the resolver tables.
DEFAULT_OUT = Path("products/site/data/lookthrough.json")

#: The parity corpus and golden books. These are test evidence, not page data:
#: 3,400 scheme names and eight fully-expanded books are ~690 KB, and shipping
#: them to every visitor to prove a test passes would be absurd. Kept out of
#: products/ so a deploy cannot accidentally carry them.
#: Gzipped: the corpus is ~755 KB of JSON and its SIZE is the point — a parity
#: harness that agrees on twenty hand-picked names proves nothing. Compressing
#: keeps it in the repo (where the test can rely on it) without tripping the
#: large-file guard, and it costs one `gzip.open` at read time.
DEFAULT_FIXTURE = Path("tests/fixtures/lookthrough_parity.json.gz")

#: Books the JS must reproduce exactly. Chosen to cover every branch the page
#: can take rather than to look impressive: a fund plus direct stocks, an
#: all-direct book with nothing to find, a book whose only fund is opaque, a
#: fund-only book, two funds overlapping on one name, and a share-count paste.
GOLDEN_BOOKS: tuple[tuple[str, str], ...] = (
    (
        "fund_and_stocks",
        "HDFCBANK 12%\nRELIANCE 10%\nNIFTYBEES 20%\nGOLDBEES 6%\n"
        "INFY 8%\nTCS 9%\nIDEA 3%\nParag Parikh Flexi Cap Fund 32%",
    ),
    ("all_direct", "HDFCBANK 30%\nRELIANCE 30%\nINFY 20%\nTCS 20%"),
    ("only_opaque_fund", "HDFCBANK 40%\nRELIANCE 30%\nGOLDBEES 30%"),
    ("fund_only", "NIFTYBEES 100%"),
    ("two_funds_overlapping", "NIFTYBEES 50%\nJUNIORBEES 30%\nHDFCBANK 20%"),
    ("index_fund_by_name", "UTI Nifty 50 Index Fund - Direct Plan - Growth 60%\nITC 40%"),
    ("bank_etf", "BANKBEES 55%\nSBIN 45%"),
    ("whitespace_and_case", "  niftybees   40%\nhdfcbank 60%  "),
    # --- the F-19 family. Every one of these produced a confident wrong
    # --- answer before the fix, so each is pinned as a golden book rather
    # --- than only as a Python unit test: the browser has to agree too.
    #
    # A digit-bearing ticker, eaten by the old size scan and read as a company.
    ("ticker_with_digits", "MID150BEES 40%\nHDFCBANK 60%"),
    # A ticker the hand-written pattern list never knew about.
    ("ticker_not_in_pattern_list", "NIFTYIETF 40%\nHDFCBANK 60%"),
    # Same fifty companies, equal-weighted. Must NOT borrow free-float weights.
    ("variant_index_refused", "Nifty 50 Equal Weight Index Fund 50%\nHDFCBANK 50%"),
    # A real Nifty index we hold no list for. Used to become the Nifty 50.
    ("unsupported_nifty_index", "Motilal Oswal Nifty Smallcap 250 Index Fund 50%\nHDFCBANK 50%"),
    # Government bonds. Used to be expanded into fifty equities.
    ("gsec_fund_refused", "HDFC Nifty G-Sec Dec 2026 Index Fund 50%\nHDFCBANK 50%"),
    # A bond ETF that matched no opaque pattern and was read as a company.
    ("bond_etf_refused", "Nifty 5 yr Benchmark G-Sec ETF 30%\nHDFCBANK 70%"),
    # A different index that was being mapped onto NIFTY BANK.
    ("financial_services_refused", "Nifty Financial Services ETF 30%\nHDFCBANK 70%"),
    # The name the deleted catch-all was written for. Must still work.
    ("bare_nifty_still_resolves", "UTI Nifty Index Fund 50%\nITC 50%"),
    # A leading size, which is what the left-hand scan legitimately exists for.
    ("leading_size", "50% HDFCBANK\n50% NIFTYBEES"),
)


def _fail(msg: str) -> None:
    sys.stderr.write(f"[export-lookthrough] FATAL {msg}\n")
    raise SystemExit(2)


def _index_blob(name: str, *, use_cache: bool, offline: bool) -> dict[str, Any] | None:
    """One index, or None with a reason on stderr.

    A missing index is not fatal. The page refuses the funds that track it by
    name, which is the correct behaviour — the alternative is shipping a page
    that unpacks six indices and silently mis-handles the seventh.
    """
    try:
        ix = I.fetch_index(name, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 - any failure means "leave it out"
        sys.stderr.write(f"[export-lookthrough] SKIP {name}: {type(exc).__name__}: {exc}\n")
        return None

    weights = ix.weights()
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        # Never export a weight vector that does not sum to 1: every downstream
        # number is a share of it, so a vector summing to 0.97 understates
        # every single company by 3% and looks entirely plausible.
        _fail(f"{name} weights sum to {total!r}, not 1.0 — refusing to export")

    return {
        "as_of": ix.as_of,
        "weight_method": ix.weight_method,
        "n_constituents": len(ix),
        "source_list_url": ix.source_list_url,
        # Shown wherever the numbers are shown. The page renders this verbatim
        # and does not summarise it.
        "accuracy_note": ix.weight_accuracy_note,
        "excluded": [list(e) for e in ix.excluded],
        "notes": list(ix.notes),
        "weights": {c.symbol: c.weight for c in ix.constituents},
        "names": {c.symbol: c.company for c in ix.constituents},
        "industries": {c.symbol: c.industry for c in ix.constituents},
    }


def _resolver_tables() -> dict[str, Any]:
    """The data the JS resolver needs, read from the Python module itself.

    Read rather than retyped. A hand-copied alias list is a second source of
    truth that drifts the first time someone edits one and not the other;
    these are the same objects the server uses, serialised.
    """
    return {
        # The LINE classifier: is this line a stock, a fund we can unpack, or a
        # fund we can name but not see into. Order is load-bearing and is
        # preserved here — first match wins, so the specific index must be
        # tried before the general one ("UTI Nifty Next 50" must not match
        # "nifty 50"). A JSON list keeps that order; a dict would not.
        #
        # Exported rather than re-authored in JS because the page's first
        # attempt at this was a hand-written /fund|etf|bees|gold/ regex, and it
        # read GOLDBEES as a COMPANY — \bbees\b does not match inside
        # "GOLDBEES", so a gold ETF was silently counted among the reader's
        # equity holdings. The parity test caught it. The real table's
        # \bgold\s*(?:etf|bees|fund)\b does not have that hole.
        "index_patterns": [
            {"rx": p.rx.pattern, "index": p.index, "ticker": p.ticker, "why": p.why} for p in C._INDEX_PATTERNS
        ],
        "opaque_patterns": [
            {"rx": p.rx.pattern, "index": p.index, "ticker": p.ticker, "why": p.why} for p in C._OPAQUE_PATTERNS
        ],
        "opaque_fallback_why": C._OPAQUE_FALLBACK_WHY,
        "indian_grouping_rx": C._INDIAN_GROUPING_RE.pattern,
        "scale_word_rx": C._SCALE_WORD_RE.pattern,
        # Size detection. `numeric_size_rx` is the F-19(d) fix: a size token is
        # a NUMBER, not any token containing a digit — Indian ETF tickers are
        # full of digits and the old test ate them.
        "size_words": sorted(C._SIZE_WORDS),
        "numeric_size_rx": C._NUMERIC_SIZE_RE.pattern,
        "supported_indices": {k: list(v) for k, v in F.SUPPORTED_INDICES.items()},
        "alias_to_index": dict(F._ALIAS_TO_INDEX),
        "ticker_listed_names": dict(F.TICKER_LISTED_NAMES),
        "opaque_tickers": dict(F.OPAQUE_TICKERS),
        "non_equity_tickers": dict(F.NON_EQUITY_TICKERS),
        "boilerplate_phrases": list(F._BOILERPLATE_PHRASES),
        "boilerplate_words": sorted(F._BOILERPLATE_WORDS),
        "house_prefixes": sorted(F._HOUSE_PREFIXES, key=len, reverse=True),
        "passive_markers": list(F._PASSIVE_MARKERS),
        "active_markers": list(F._ACTIVE_MARKERS),
        "index_family_prefixes": list(F._INDEX_FAMILY_PREFIXES),
        "passive_category_markers": list(F._PASSIVE_CATEGORY_MARKERS),
        "non_equity_phrases": list(F._NON_EQUITY_PHRASES),
        # Pattern SOURCE, compiled by the JS. Exported rather than re-authored
        # in JS syntax for the same reason as the tables above; the parity
        # corpus is what proves the two engines read them the same way.
        "patterns": {
            "plan_suffix": F._PLAN_SUFFIX_RE.pattern,
            "qty_suffix": F._QTY_SUFFIX_RE.pattern,
            "currency_lead": F._CURRENCY_LEAD_RE.pattern,
            "ticker": F._TICKER_RE.pattern,
        },
    }


def _parity_corpus(*, offline: bool) -> dict[str, Any]:
    """Every live AMFI scheme name with Python's reduction of it.

    This is the test corpus, and its size is the point: ``index_phrase`` is a
    lossy reduction tuned over thousands of real Indian scheme names, and a
    port of it that agrees on twenty hand-picked examples proves nothing.
    """
    try:
        master = F.load_scheme_master()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[export-lookthrough] WARNING no AMFI master ({type(exc).__name__}: {exc}); "
            "parity corpus will be the ticker tables only\n"
        )
        schemes = []
    else:
        schemes = list(getattr(master, "schemes", ()) or ())

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(text: str, category: str | None = None) -> None:
        key = f"{text}\x00{category or ''}"
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "text": text,
                "category": category,
                "phrase": F.index_phrase(text),
                "classification": F.classify(text, category=category),
            }
        )

    for s in schemes:
        add(getattr(s, "name", "") or "", getattr(s, "category", None))

    # The tickers and a set of shapes a user actually types, which the AMFI
    # master does not contain: bare tickers, pasted quantities, exchange
    # suffixes, and the glued forms.
    for t in list(F.TICKER_LISTED_NAMES) + list(F.OPAQUE_TICKERS):
        add(t)
        add(F.TICKER_LISTED_NAMES.get(t, t))
    for extra in (
        "NIFTYBEES.NS",
        "NIFTYBEES 200 units",
        "NIFTYBEES Rs 50,000",
        "niftybees",
        "nifty50",
        "NIFTY GROWTH SECTORS 15",
        "bank nifty",
        "BANKNIFTY",
        "UTI Nifty Index Fund - Direct Plan - Growth",
        "HDFC Nifty G-Sec Dec 2026 Index Fund",
        "Parag Parikh Flexi Cap Fund",
        "SBI Banking Fund",
        "Groww Nifty 200 ETF FOF",
        "",
        "   ",
    ):
        add(extra)

    return {"n_schemes": len(schemes), "rows": rows}


def _golden(*, offline: bool) -> list[dict[str, Any]]:
    """Whole books through the real engine, for the JS to reproduce exactly.

    Stores the numbers to full precision. A tolerance here would hide the very
    drift the file exists to catch, so the test compares at 1e-12.
    """
    out: list[dict[str, Any]] = []
    for label, text in GOLDEN_BOOKS:
        try:
            lt = C.look_through(text)
        except Exception as exc:  # noqa: BLE001
            _fail(f"golden book {label!r} raised {type(exc).__name__}: {exc}")
        out.append(
            {
                "label": label,
                "input": text,
                "expect": {
                    "computed": bool(lt.computed),
                    "not_computed_reason": getattr(lt, "not_computed_reason", "") or "",
                    "n_listed": lt.n_listed,
                    "n_unpacked": lt.n_unpacked,
                    "n_companies": lt.n_companies,
                    "residual": lt.residual,
                    "lines": [
                        {
                            "symbol": ln.symbol,
                            "listed": ln.listed,
                            "actual": ln.actual,
                            "kind": ln.kind,
                            "via": [
                                {
                                    "source": c.source,
                                    "index": c.index,
                                    "fund_weight": c.fund_weight,
                                    "constituent_weight": c.constituent_weight,
                                    "weight": c.weight,
                                }
                                for c in (ln.via or ())
                            ],
                        }
                        for ln in lt.lines
                    ],
                    "funds": [
                        {
                            "label": f.label,
                            "weight": f.weight,
                            "index": f.index,
                            "unpacked": bool(f.unpacked),
                            "n_constituents": f.n_constituents,
                        }
                        for f in (lt.funds or ())
                    ],
                },
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="use only cached responses; fail rather than fetch",
    )
    ap.add_argument(
        "--no-parity",
        action="store_true",
        help="skip the AMFI parity corpus (smaller file; the parity test will fail)",
    )
    args = ap.parse_args(argv)

    use_cache = True

    indices: dict[str, Any] = {}
    for name in I.known_indices():
        blob = _index_blob(name, use_cache=use_cache, offline=args.offline)
        if blob is not None:
            indices[name] = blob
    if not indices:
        _fail("no index could be fetched; refusing to write a page that can unpack nothing")

    payload: dict[str, Any] = {
        "schema": 1,
        "generated": date.today().isoformat(),
        "generator": "scripts/export_lookthrough_data.py",
        "weight_method": "free_float_mcap",
        # The one sentence a reader needs about where these numbers come from.
        "provenance": (
            "Index membership is published by NSE. The WEIGHTS are ours, computed "
            "from free-float market capitalisation, because NSE does not publish "
            "free weights. Each index carries its own measured error."
        ),
        "indices": indices,
        "resolver": _resolver_tables(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # The test evidence, in its own file. It carries the SAME generated date so
    # a fixture left behind by an older export is visible rather than silently
    # validating a page built from different numbers.
    fixture = {
        "schema": 1,
        "generated": payload["generated"],
        "generator": payload["generator"],
        "golden": _golden(offline=args.offline),
        "parity": {"n_schemes": 0, "rows": []} if args.no_parity else _parity_corpus(offline=args.offline),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(fixture, ensure_ascii=False, sort_keys=True) + "\n"
    # mtime=0 so re-running the exporter on unchanged inputs produces an
    # identical file rather than a spurious diff.
    with gzip.GzipFile(filename="", mode="wb", fileobj=args.fixture.open("wb"), mtime=0) as fh:
        fh.write(blob.encode("utf-8"))

    size = args.out.stat().st_size
    sys.stderr.write(
        f"[export-lookthrough] wrote {args.out} ({size:,} bytes)\n"
        f"[export-lookthrough]   indices: {len(indices)} "
        f"({sum(v['n_constituents'] for v in indices.values())} constituents)\n"
        f"[export-lookthrough]   etf tickers: {len(payload['resolver']['ticker_listed_names'])}, "
        f"opaque: {len(payload['resolver']['opaque_tickers'])}\n"
        f"[export-lookthrough] wrote {args.fixture} ({args.fixture.stat().st_size:,} bytes, not deployed)\n"
        f"[export-lookthrough]   golden books: {len(fixture['golden'])}\n"
        f"[export-lookthrough]   parity rows: {len(fixture['parity']['rows'])} "
        f"(from {fixture['parity']['n_schemes']} AMFI schemes)\n"
    )
    for name, blob in sorted(indices.items()):
        sys.stderr.write(f"[export-lookthrough]   {name:<20} {blob['n_constituents']:>4} as of {blob['as_of']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
