---
title: "Research: DeFi Protocol On-Chain Flows (7b-L)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/defi
---

# Research: DeFi Protocol On-Chain Flows (7b-L)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/defi_flows.py` → `DefiFlowsTool`
**Status:** IMPLEMENTED, TESTED

## APIs Probed

### DefiLlama — Protocols ✅ SELECTED
- **URL:** `https://api.llama.fi/protocols`
- **Method:** GET
- **Auth:** None
- **Format:** JSON (array of 7000+ protocols)
- **Rate limits:** None documented, generous
- **Coverage:** **Global** — all EVM/non-EVM chains. Borderless blockchain data.

### DefiLlama — Stablecoins ✅ SELECTED
- **URL:** `https://stablecoins.llama.fi/stablecoins?includePrices=true`
- **Method:** GET
- **Format:** JSON (350+ stablecoins, supply by chain)

### DefiLlama — DEX Volume ✅ SELECTED
- **URL:** `https://api.llama.fi/overview/dexs?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true&dataType=dailyVolume`
- **Method:** GET
- **Format:** JSON (1000+ DEXes, 24h volume)

## Geographic Coverage
- Blockchain is inherently borderless — no geographic boundaries
- Protocols span Ethereum, Solana, BSC, Arbitrum, Polygon, Avalanche, etc.
- **Verdict:** `[G:GLOBAL]`

## Modes Implemented
1. `tvl` — top protocols by Total Value Locked, with market category filtering
2. `stablecoins` — supply by issuer (USDT, USDC, DAI, etc.)
3. `dex_volume` — 24h DEX trading volumes
4. `chain` — TVL aggregated by blockchain

## Signal Value
- TVL drops = capital flight from DeFi (risk-off)
- Stablecoin supply shifts = flight-to-safety vs risk-on rotation
- DEX volume surges = volatility spike, panic trading
- Chain migration = ecosystem health indicator

## Risks
- DefiLlama is a community project — could change API without notice
- TVL can be inflated by double-counting across chains
- No historical API in current modes — snapshot only

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]
