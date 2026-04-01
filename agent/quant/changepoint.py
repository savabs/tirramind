"""Bayesian Online Changepoint Detection (Adams & MacKay 2007).

Detects abrupt changes in the generative parameters of a sequential data stream.
Uses a Normal observation model with conjugate Normal-Inverse-Gamma prior and
constant hazard function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass
class BOCPDResult:
    """Result of BOCPD inference."""

    run_length_posterior: np.ndarray  # (T, max_rl) — P(r_t | x_{1:t})
    changepoint_probs: np.ndarray    # (T,) — P(changepoint at t)

    @property
    def map_run_lengths(self) -> np.ndarray:
        """Most probable run length at each timestep."""
        return np.argmax(self.run_length_posterior, axis=1)

    @property
    def expected_run_lengths(self) -> np.ndarray:
        """Expected run length at each timestep."""
        T, max_rl = self.run_length_posterior.shape
        r_vals = np.arange(max_rl)
        return self.run_length_posterior @ r_vals

    def changepoints(self, min_drop_frac: float = 0.5, min_prev_rl: int = 20) -> list[int]:
        """Detect changepoints via expected run-length drops.

        A changepoint is detected at time t when:
        1. The expected run length drops by more than `min_drop_frac` of
           its previous value.
        2. The previous expected run length was at least `min_prev_rl`.
        3. The new expected run length is less than `min_prev_rl` (to filter
           truncation artifacts where E[rl] drops but stays high).

        Returns list of timestep indices where changepoints were detected.
        """
        erl = self.expected_run_lengths
        cps = []
        for t in range(1, len(erl)):
            prev = erl[t - 1]
            curr = erl[t]
            if (
                prev >= min_prev_rl
                and curr < prev * (1 - min_drop_frac)
                and curr < min_prev_rl
            ):
                cps.append(t)
        return cps


class BOCPD:
    """Bayesian Online Changepoint Detection.

    Parameters
    ----------
    hazard_lambda : float
        Expected run length (mean time between changepoints).
        Hazard = 1/hazard_lambda (constant).
    prior_mu : float
        Prior mean for the Normal-Inverse-Gamma conjugate.
    prior_kappa : float
        Prior pseudo-count for the mean.
    prior_alpha : float
        Prior shape for the Inverse-Gamma on variance.
    prior_beta : float
        Prior scale for the Inverse-Gamma on variance.
    """

    def __init__(
        self,
        hazard_lambda: float = 200,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.hazard_lambda = hazard_lambda
        self.H = 1.0 / hazard_lambda  # constant hazard
        self.mu0 = prior_mu
        self.kappa0 = prior_kappa
        self.alpha0 = prior_alpha
        self.beta0 = prior_beta

    def fit(self, data: np.ndarray) -> BOCPDResult:
        """Run BOCPD on a 1-D time series.

        Parameters
        ----------
        data : 1-D array of length T.

        Returns
        -------
        BOCPDResult with run-length posterior and changepoint probabilities.
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        T = len(data)
        max_rl = int(2 * self.hazard_lambda)  # truncation bound

        # Sufficient statistics arrays (one per possible run length)
        # NIG posterior: mu, kappa, alpha, beta updated sequentially
        mu = np.full(max_rl + 1, self.mu0)
        kappa = np.full(max_rl + 1, self.kappa0)
        alpha = np.full(max_rl + 1, self.alpha0)
        beta = np.full(max_rl + 1, self.beta0)

        # Run-length distribution: R[r] = P(r_t = r | x_{1:t})
        # At t=0, run length is 0 with probability 1
        R = np.zeros(max_rl + 1)
        R[0] = 1.0

        # Storage
        rl_posterior = np.zeros((T, max_rl + 1))
        cp_probs = np.zeros(T)

        for t in range(T):
            x = data[t]

            # 1. Evaluate predictive probability for each run length
            #    Student-t: p(x | mu, kappa, alpha, beta)
            pred = self._student_t_pdf(x, mu, kappa, alpha, beta)

            # 2. Growth probabilities: P(r_t = r+1, x_{1:t})
            joint_grow = R * pred * (1 - self.H)

            # 3. Changepoint probability: P(r_t = 0, x_{1:t})
            joint_cp = np.sum(R * pred * self.H)

            # 4. Shift run lengths (grow by 1) and set r=0
            new_R = np.zeros_like(R)
            new_R[1 : max_rl + 1] = joint_grow[:max_rl]
            new_R[0] = joint_cp

            # 5. Normalize
            evidence = new_R.sum()
            if evidence > 0:
                new_R /= evidence

            # 6. Update sufficient statistics for the NIG posterior
            #    Shift stats to account for r → r+1
            new_mu = np.empty_like(mu)
            new_kappa = np.empty_like(kappa)
            new_alpha = np.empty_like(alpha)
            new_beta = np.empty_like(beta)

            # r=0: reset to prior (new segment)
            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            # r>0: Bayesian update of NIG with observation x
            old_mu = mu[:max_rl]
            old_kappa = kappa[:max_rl]
            old_alpha = alpha[:max_rl]
            old_beta = beta[:max_rl]

            new_kappa[1:] = old_kappa + 1
            new_mu[1:] = (old_kappa * old_mu + x) / new_kappa[1:]
            new_alpha[1:] = old_alpha + 0.5
            new_beta[1:] = (
                old_beta
                + 0.5 * old_kappa * (x - old_mu) ** 2 / new_kappa[1:]
            )

            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
            R = new_R

            # Store
            rl_posterior[t] = R
            cp_probs[t] = R[0]  # P(r_t=0) = probability of changepoint at t

        return BOCPDResult(
            run_length_posterior=rl_posterior,
            changepoint_probs=cp_probs,
        )

    @staticmethod
    def _student_t_pdf(
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Predictive probability under the NIG model (Student-t).

        p(x | mu, kappa, alpha, beta) = Student-t with:
            df = 2*alpha
            loc = mu
            scale = beta*(kappa+1) / (alpha*kappa)
        """
        df = 2 * alpha
        scale = beta * (kappa + 1) / (alpha * kappa)

        # Student-t PDF (vectorized, avoids deprecated np.lgamma)
        z = (x - mu) ** 2 / scale
        log_coeff = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df * scale)
        )
        log_pdf = log_coeff - ((df + 1) / 2) * np.log(1 + z / df)
        return np.exp(log_pdf)
