---
title: "Task: math_stack_roadmap"
tags: [doc/task, topic/math-stack, status/active]
---

# Math Stack Roadmap

Created: 2026-05-29 | Related: [[quant_training_ground]], [[agent_playground_doctrine]]

> Math stack serves **Layer 2** for **N1+N4 playground** ([[n1_n4_playground_spec]]). **M9 microstructure is priority** (micro-changes decide outcomes). No sentiment features — mathematical field over finance.

**2026-06-03 strategy:** [[full_quant_base_model]] — LLM-style **quant base pretrain** (M14+M15 on graph) → **N1+N4 finetune**. M14/M15 are the critical path; M1–M6 libraries exist but are not on the GNN yet. Task: [[full_quant_base_model_task]].

## M1: Differentiable SDE Foundation ✅ COMPLETE (2026-05-30)
- [x] M1.1: Integrate `torchsde`, verify GPU + autograd
- [x] M1.2: Euler-Maruyama + Milstein via torchsde.sdeint
- [x] M1.3: GBM module (learnable μ, σ)
- [x] M1.4: Heston SDE (2-factor: price + variance)
- [x] M1.5: Tests — 15/15 pass (grad flows, GPU, adjoint, frozen params)

## M2: Options Pricing & Greeks ✅ COMPLETE (2026-05-31)
- [x] M2.1: Differentiable Black-Scholes (European) in PyTorch
- [x] M2.2: Greeks via `torch.autograd.grad` (delta, gamma, vega, theta, rho)
- [x] M2.3: Implied vol solver (Newton-Raphson, differentiable)
- [x] M2.4: American options (Barone-Adesi-Whaley, differentiable)
- [x] M2.5: Test: Greeks match analytical to 1e-5, 11/11 tests pass

## M3: Advanced Pricing Models
- [x] M3.1: Heston pricing (COS method, differentiable)
- [x] M3.2: Bates (Heston + jumps)
- [x] M3.3: Merton jump-diffusion
- [x] M3.4: Variance Gamma / NIG Levy
- [x] M3.5: Test: vs Monte Carlo benchmark

## M4: Fourier Pricing
- [ ] M4.1: Carr-Madan FFT pricing
- [x] M4.2: COS method (Fang & Oosterlee 2008)
- [ ] M4.3: ML tuning parameter selection (arXiv:2412.05070)
- [ ] M4.4: Convolution-FFT (arXiv:2512.05326)

## M5: Implied Volatility Surface
- [x] M5.1: SVI parameterization (Gatheral 2014), no-arbitrage constraints
- [x] M5.2: SABR model via `pysabr` (vectorized, differentiable PyTorch implementation)
- [x] M5.3: Surface features → GNN instrument nodes (ATM vol, skew, curvature, term structure, SVI params)
- [x] M5.4: Test: no calendar/butterfly arbitrage

## M6: Rough Volatility
- [x] M6.1: Rough Bergomi (Bayer, Friz, Gatheral)
- [x] M6.2: Hybrid simulation (Bennedsen, Lunde, Pakkanen)
- [ ] M6.3: Rough Heston / lifted Heston (Abi Jaber 2019)
- [ ] M6.4: Deep calibration via NN (Bayer & Stemper)
- [x] M6.5: Hurst exponent estimation (H < 0.5 = rough)
- [x] M6.6: Test: ATM skew ~ T^{H-1/2}

## M7: Portfolio Optimization
- [ ] M7.1: Mean-variance (differentiable, PyTorch)
- [ ] M7.2: CVaR optimization (Rockafellar-Uryasev, differentiable)
- [ ] M7.3: Risk parity / equal risk contribution
- [ ] M7.4: Hierarchical risk parity (Lopez de Prado)
- [ ] M7.5: Entropy pooling (Meucci) for stress-testing
- [ ] M7.6: Black-Litterman Bayesian allocation
- [ ] M7.7: Diffable portfolio head: GNN→return dist→CVaR opt→weights (end-to-end gradient)
- [ ] M7.8: Test: CVaR < VaR, weights sum=1, turnover constraints

## M8: RL Derivative Hedging
- [ ] M8.1: Hedging env (option position, underlying, costs, limits)
- [ ] M8.2: SAC agent (arXiv:2512.12420, Dec 2025)
- [ ] M8.3: GNN embeddings as state representation
- [ ] M8.4: Multi-asset hedging
- [ ] M8.5: Benchmark vs delta/gamma/no-hedge
- [ ] M8.6: Test: RL Sharpe > delta hedge Sharpe

## M9: Market Microstructure — ✅ COMPLETE (2026-06-01)
- [x] M9.1: OFI computation
- [x] M9.2: VPIN (informed trading probability)
- [ ] M9.3: Hawkes OFI forecasting (arXiv:2408.03594) — deferred to Phase 2 (requires tick data)
- [x] M9.4: Bid-ask spread (Roll, Corwin-Schultz)
- [x] M9.5: Kyle's lambda (price impact)
- [x] M9.6: Microstructure features → GNN inputs
- [ ] M9.7: Test: VPIN spike before known events — deferred to Phase 2 (requires real data)

## M10: Information Theory & Causality
- [ ] M10.1: Transfer entropy (Schreiber 2000) — directional information flow
- [ ] M10.2: Mutual information (non-linear dependence, k-NN estimator)
- [ ] M10.3: Rényi transfer entropy (generalized, α parameter)
- [ ] M10.4: Feature selection via MI ranking
- [ ] M10.5: Causal graph edge weighting via TE
- [ ] M10.6: Test: TE detects known causal direction in synthetic data

## M11: Random Matrix Theory
- [ ] M11.1: Marcenko-Pastur spectral density fitting
- [ ] M11.2: Eigenvalue cleaning (denoise correlation matrix)
- [ ] M11.3: Signal vs noise separation (eigenvalues > MP upper bound)
- [ ] M11.4: Test: cleaned matrix improves portfolio out-of-sample Sharpe

## M12: Information Geometry
- [ ] M12.1: Fisher information metric on model parameter space
- [ ] M12.2: Natural gradient descent (Fisher-preconditioned)
- [ ] M12.3: Test: natural gradient converges faster on toy problem

## M13: Malliavin Calculus
- [ ] M13.1: Malliavin weights for discontinuous payoffs
- [ ] M13.2: Greeks via Malliavin integration-by-parts (when pathwise fails)
- [ ] M13.3: Test: digital option Greeks match likelihood ratio method

## M14: GNN Integration (Cross-Cutting)
- [ ] M14.1: Options/Greeks/IV features as instrument node features
- [ ] M14.2: Tail-dependence (copula) as edge weights
- [ ] M14.3: Transfer entropy as directed edge weights
- [ ] M14.4: Portfolio weights as GNN prediction head
- [ ] M14.5: RL hedging agent reads GNN embeddings as state
- [ ] M14.6: End-to-end test: raw data → GNN → pricing → portfolio → P&L gradient

## M15: Data Requirements
- [ ] M15.1: Options chain data (SPX/SPY minimum, ideally all liquid underlyings)
- [ ] M15.2: Order book / tick data (for microstructure)
- [ ] M15.3: Interest rate curve (for pricing)
- [ ] M15.4: Dividend data (for BSM)

---

## Key Libraries to Adopt
| Domain | Library |
|--------|---------|
| SDE | `torchsde` (Google) |
| Options/Greeks | `vollib`, `tf-quant-finance` |
| IV Surface | SVI + `pysabr` |
| Rough Vol | `rough_bergomi` (McCrickerd) |
| Portfolio | `Riskfolio-Lib` |
| RL Hedging | Custom (arXiv:2512.12420) |
| Fourier | COS method + ML tuning |
