"""Hidden Markov Model regime detection via hmmlearn."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeResult:
    """Result of HMM regime fitting."""

    states: np.ndarray  # (T,) — state labels per timestep
    means: np.ndarray  # (K,) — state means (sorted ascending)
    variances: np.ndarray  # (K,) — state variances
    transition_matrix: np.ndarray  # (K, K) — row-stochastic
    log_likelihood: float

    @property
    def n_states(self) -> int:
        return len(self.means)


class RegimeHMM:
    """Gaussian HMM for regime detection.

    States are relabeled after fitting so that state 0 has the lowest
    mean (contraction) and state K-1 the highest (expansion).

    Parameters
    ----------
    n_states : int
        Number of hidden states (regimes).
    n_init : int
        Number of random initializations for EM (best kept).
    max_iter : int
        Maximum EM iterations per initialization.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_init: int = 10,
        max_iter: int = 100,
    ) -> None:
        self.n_states = n_states
        self.n_init = n_init
        self.max_iter = max_iter
        self._model: GaussianHMM | None = None

    def fit(self, data: np.ndarray) -> RegimeResult:
        """Fit the HMM to a 1-D time series.

        Parameters
        ----------
        data : 1-D array of length T.

        Returns
        -------
        RegimeResult with state labels, means, variances, transitions, LL.
        """
        X = np.asarray(data, dtype=np.float64).reshape(-1, 1)

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.max_iter,
            random_state=42,
        )

        # Run multiple initializations, keep best
        best_model = None
        best_ll = -np.inf
        for _ in range(self.n_init):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X)
                    ll = model.score(X)
                    if ll > best_ll:
                        best_ll = ll
                        # Deep copy the fitted model params
                        best_model = GaussianHMM(
                            n_components=self.n_states,
                            covariance_type="full",
                            n_iter=self.max_iter,
                        )
                        best_model.startprob_ = model.startprob_.copy()
                        best_model.transmat_ = model.transmat_.copy()
                        best_model.means_ = model.means_.copy()
                        best_model.covars_ = model.covars_.copy()
                except Exception:
                    continue
            # Re-randomize for next init
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.max_iter,
                random_state=None,
            )

        if best_model is None:
            raise RuntimeError("HMM fitting failed on all initializations")

        self._model = best_model
        states = best_model.predict(X)
        return self._build_result(best_model, states, best_ll)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Classify new data using fitted model (frozen params).

        Returns 1-D array of state labels.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        X = np.asarray(data, dtype=np.float64).reshape(-1, 1)
        raw_states = self._model.predict(X)
        # Apply same relabeling as fit
        order = np.argsort(self._model.means_.ravel())
        label_map = np.zeros(self.n_states, dtype=int)
        for new_label, old_label in enumerate(order):
            label_map[old_label] = new_label
        return label_map[raw_states]

    def _build_result(self, model: GaussianHMM, raw_states: np.ndarray, ll: float) -> RegimeResult:
        """Relabel states by ascending mean and build result."""
        raw_means = model.means_.ravel()
        raw_vars = model.covars_.reshape(self.n_states, -1)[:, 0]  # diagonal elements
        raw_trans = model.transmat_

        # Sort states by ascending mean
        order = np.argsort(raw_means)
        sorted_means = raw_means[order]
        sorted_vars = raw_vars[order]
        sorted_trans = raw_trans[np.ix_(order, order)]

        # Relabel state sequence
        label_map = np.zeros(self.n_states, dtype=int)
        for new_label, old_label in enumerate(order):
            label_map[old_label] = new_label
        sorted_states = label_map[raw_states]

        return RegimeResult(
            states=sorted_states,
            means=sorted_means,
            variances=sorted_vars,
            transition_matrix=sorted_trans,
            log_likelihood=ll,
        )
