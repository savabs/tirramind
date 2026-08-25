"""
M9 Market Microstructure Feature Extractors

Differentiable PyTorch implementations of microstructure indicators:
- M9.1: Order Flow Imbalance (OFI)
- M9.2: VPIN (Volume-Synchronized Probability of Informed Trading)
- M9.4: Bid-Ask Spread (Corwin-Schultz, Roll)
- M9.5: Kyle's Lambda (Price Impact)

References:
- Cont, Kukanov, Stoikov (2014). "The Price Impact of Order Book Events." JFE
- Easley, López de Prado, O'Hara (2012). "Flow Toxicity and Liquidity." RFS
- Corwin & Schultz (2012). "A Simple Way to Estimate Bid-Ask Spreads." JoF
- Kyle (1985). "Continuous Auctions and Insider Trading." Econometrica
- Nittur Anantha & Jain (2024). "Forecasting High Frequency OFI." arXiv:2408.03594

Doctrine:
- Differentiability invariant: all forward passes in PyTorch
- Signal Depth L2: per-instrument features
- No sentiment, mathematical field over finance
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math


class SpreadEstimator(nn.Module):
    """Bid-ask spread estimation from OHLC data (M9.4)."""
    
    def __init__(self):
        super().__init__()
        
    def corwin_schultz(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        variant: str = "CSD"
    ) -> torch.Tensor:
        """
        Corwin-Schultz spread estimator from daily high/low prices.
        
        Args:
            high: (T,) daily high prices
            low: (T,) daily low prices
            variant: "CSD" (daily censored), "CSM" (censored mean), "CSP" (positive only)
            
        Returns:
            (T-1,) daily spread estimates
        """
        high = torch.clamp(high, min=1e-8)
        low = torch.clamp(low, min=1e-8)
        
        beta = torch.log(high / low).pow(2)
        
        high_2day = torch.maximum(high[:-1], high[1:])
        low_2day = torch.minimum(low[:-1], low[1:])
        gamma = torch.log(high_2day / low_2day).pow(2)
        
        sqrt_beta = torch.sqrt(beta[:-1])
        sqrt_2beta = torch.sqrt(2.0 * beta[:-1])
        sqrt_gamma = torch.sqrt(gamma)
        
        denom = 3.0 - 2.0 * math.sqrt(2.0)
        alpha = (sqrt_2beta - sqrt_beta) / denom - torch.sqrt(gamma / denom)
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        
        exp_alpha = torch.exp(alpha)
        spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
        
        if variant == "CSD":
            spread = torch.clamp(spread, min=0.0)
        
        spread = torch.nan_to_num(spread, nan=0.0)
        return spread
    
    def roll_measure(self, returns: torch.Tensor) -> torch.Tensor:
        """Roll (1984) spread estimator from return autocovariance."""
        if len(returns) < 2:
            return torch.tensor(0.0, device=returns.device)
        
        mean_return = returns.mean()
        centered = returns - mean_return
        cov = (centered[:-1] * centered[1:]).mean()
        
        if cov < 0:
            spread = 2.0 * torch.sqrt(-cov)
        else:
            spread = torch.tensor(0.0, device=returns.device)
        
        return spread


class OrderFlowImbalance(nn.Module):
    """Order Flow Imbalance computation (M9.1)."""
    
    def __init__(self):
        super().__init__()
        
    def compute_ofi_trade_tape(
        self,
        prices: torch.Tensor,
        volumes: torch.Tensor,
        window_size: int = 12
    ) -> torch.Tensor:
        """Compute OFI using BVC-style price change classification."""
        T = len(prices)
        ofi = torch.zeros(T, device=prices.device)
        
        for t in range(window_size, T):
            window_prices = prices[t-window_size:t]
            window_volumes = volumes[t-window_size:t]
            
            p_open = window_prices[0]
            p_close = window_prices[-1]
            delta_p = p_close - p_open
            
            sigma = torch.std(window_prices)
            if sigma < 1e-8:
                sigma = torch.tensor(1e-8, device=prices.device)
            
            z = delta_p / sigma
            prob_buy = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            
            total_volume = window_volumes.sum()
            v_buy = total_volume * prob_buy
            v_sell = total_volume * (1.0 - prob_buy)
            
            ofi[t] = v_buy - v_sell
        
        return ofi
    
    def compute_ofi_zscore(self, ofi: torch.Tensor, window: int = 720) -> torch.Tensor:
        """Standardize OFI to z-score vs. trailing distribution."""
        T = len(ofi)
        ofi_zscore = torch.zeros(T, device=ofi.device)
        
        for t in range(window, T):
            window_ofi = ofi[t-window:t]
            mean = window_ofi.mean()
            std = window_ofi.std()
            
            if std < 1e-8:
                std = torch.tensor(1e-8, device=ofi.device)
            
            ofi_zscore[t] = (ofi[t] - mean) / std
        
        return ofi_zscore


@dataclass
class VolumeBucket:
    """Container for volume-bucketed trade data."""
    open_price: float
    close_price: float
    high: float
    low: float
    total_volume: float
    timestamp_end: int


class VPINCalculator(nn.Module):
    """VPIN (Volume-Synchronized Probability of Informed Trading) (M9.2)."""
    
    def __init__(self, bucket_volume: float = 100000.0, n_buckets: int = 50):
        super().__init__()
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        
    def bucket_trades(
        self,
        trades: List[Tuple[float, float, int]]
    ) -> List[VolumeBucket]:
        """Partition trades into equal-volume buckets."""
        buckets = []
        acc_volume = 0.0
        acc_prices = []
        start_ts = None
        
        for price, volume, timestamp in trades:
            if start_ts is None:
                start_ts = timestamp
            
            acc_volume += volume
            acc_prices.append(price)
            
            if acc_volume >= self.bucket_volume:
                bucket = VolumeBucket(
                    open_price=acc_prices[0],
                    close_price=acc_prices[-1],
                    high=max(acc_prices),
                    low=min(acc_prices),
                    total_volume=acc_volume,
                    timestamp_end=timestamp
                )
                buckets.append(bucket)
                
                acc_volume = 0.0
                acc_prices = []
                start_ts = None
        
        return buckets
    
    def classify_buckets_bvc(self, buckets: List[VolumeBucket]) -> torch.Tensor:
        """Classify each bucket volume as buy/sell using BVC."""
        N = len(buckets)
        classified = torch.zeros((N, 2))
        
        for i, bucket in enumerate(buckets):
            delta_p = bucket.close_price - bucket.open_price
            
            sigma = (bucket.high - bucket.low) / 2.0
            if sigma < 1e-8:
                sigma = 1e-8
            
            z = delta_p / sigma
            prob_buy = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            
            v_buy = bucket.total_volume * prob_buy
            v_sell = bucket.total_volume * (1.0 - prob_buy)
            
            classified[i, 0] = v_buy
            classified[i, 1] = v_sell
        
        return classified
    
    def compute_vpin(self, classified_buckets: torch.Tensor) -> torch.Tensor:
        """Compute VPIN over rolling window."""
        N = classified_buckets.shape[0]
        if N < self.n_buckets:
            return torch.tensor([0.0])
        
        vpin_values = []
        
        for t in range(self.n_buckets, N + 1):
            window = classified_buckets[t-self.n_buckets:t]
            
            v_buy = window[:, 0]
            v_sell = window[:, 1]
            
            imbalance = torch.abs(v_buy - v_sell).sum()
            total_volume = (v_buy + v_sell).sum()
            
            if total_volume > 1e-8:
                vpin = imbalance / total_volume
            else:
                vpin = torch.tensor(0.0)
            
            vpin_values.append(vpin)
        
        return torch.stack(vpin_values)


class KyleLambdaEstimator:
    """Kyle's Lambda estimation via rolling OLS (M9.5). Not a nn.Module."""
    
    def estimate_lambda(
        self,
        prices: np.ndarray,
        signed_volume: np.ndarray,
        window_hours: int = 720
    ) -> np.ndarray:
        """Rolling OLS: ΔP_t = α + λ × Q_t + ε_t"""
        T = len(prices)
        lambda_estimates = []
        
        for t in range(window_hours, T):
            window_prices = prices[t-window_hours:t]
            window_volumes = signed_volume[t-window_hours:t-1]
            
            delta_p = np.diff(window_prices)
            Q = window_volumes
            
            Q_mean = Q.mean()
            delta_p_mean = delta_p.mean()
            
            numerator = ((Q - Q_mean) * (delta_p - delta_p_mean)).sum()
            denominator = ((Q - Q_mean) ** 2).sum()
            
            if denominator > 1e-8:
                lambda_t = numerator / denominator
            else:
                lambda_t = 0.0
            
            lambda_estimates.append(lambda_t)
        
        return np.array(lambda_estimates)


class MicrostructureFeatureExtractor(nn.Module):
    """Unified extractor for all M9 microstructure features."""
    
    def __init__(self):
        super().__init__()
        self.spread_estimator = SpreadEstimator()
        self.ofi_calculator = OrderFlowImbalance()
        self.vpin_calculator = VPINCalculator()
        
    def forward(
        self,
        ohlcv: Dict[str, torch.Tensor],
        trades: Optional[List] = None,
        precomputed_lambda: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract all available M9 features."""
        features = {}
        
        if "high" in ohlcv and "low" in ohlcv:
            features["spread_cs"] = self.spread_estimator.corwin_schultz(
                ohlcv["high"], ohlcv["low"]
            )
        
        if "close" in ohlcv:
            returns = torch.log(ohlcv["close"][1:] / ohlcv["close"][:-1])
            features["spread_roll"] = self.spread_estimator.roll_measure(returns)
        
        if trades is not None:
            prices = torch.tensor([t[0] for t in trades])
            volumes = torch.tensor([t[1] for t in trades])
            
            ofi = self.ofi_calculator.compute_ofi_trade_tape(prices, volumes)
            features["ofi"] = ofi
            features["ofi_zscore"] = self.ofi_calculator.compute_ofi_zscore(ofi)
            
            buckets = self.vpin_calculator.bucket_trades(trades)
            if len(buckets) >= self.vpin_calculator.n_buckets:
                classified = self.vpin_calculator.classify_buckets_bvc(buckets)
                features["vpin"] = self.vpin_calculator.compute_vpin(classified)
            else:
                features["vpin"] = torch.tensor([0.0])
        
        if precomputed_lambda is not None:
            features["kyle_lambda"] = precomputed_lambda
            features["lambda_regime"] = self._classify_lambda_regime(precomputed_lambda)
        
        if "vpin" in features:
            features["vpin_regime"] = self._classify_vpin_regime(features["vpin"])
        
        return features
    
    def _classify_vpin_regime(self, vpin: torch.Tensor) -> torch.Tensor:
        """Classify VPIN into regimes: [normal, elevated, extreme]."""
        T = len(vpin)
        regime = torch.zeros((T, 3))
        
        for t in range(T):
            v = vpin[t]
            if v < 0.3:
                regime[t, 0] = 1.0
            elif v < 0.5:
                regime[t, 1] = 1.0
            else:
                regime[t, 2] = 1.0
        
        return regime
    
    def _classify_lambda_regime(self, lambda_ts: torch.Tensor) -> torch.Tensor:
        """Classify Kyle's lambda into regimes: [liquid, normal, illiquid]."""
        T = len(lambda_ts)
        regime = torch.zeros((T, 3))
        
        p33 = torch.quantile(lambda_ts, 0.33)
        p66 = torch.quantile(lambda_ts, 0.66)
        
        for t in range(T):
            lam = lambda_ts[t]
            if lam < p33:
                regime[t, 0] = 1.0
            elif lam < p66:
                regime[t, 1] = 1.0
            else:
                regime[t, 2] = 1.0
        
        return regime
