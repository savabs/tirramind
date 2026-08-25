"""
TirraMind — Diagonal Diffusion Head (M1 Component, Phase E)

Parameterizes the diffusion term g_phi(X) in the M1 Neural SDE:

    dX_i(t) = f_theta(X_i, G, Z) dZ(t)   [CDE drift — always active]
            + g_phi(X_i) dW_i(t)          [SDE diffusion — Phase E only]

g_phi(X_i) returns a per-dimension positive noise scale, implemented as a
2-layer MLP with softplus output activation and a minimum noise floor that
prevents KL collapse.

During training: the noise eps ~ N(0,I) is sampled, scaled by g_phi(X_i)
and sqrt(dt), and added to the state after each Euler-Maruyama drift step.

During inference: g_phi(X_i) is computed but noise is NOT added.  The output
serves only as an uncertainty estimate (per-dimension confidence interval
for the entity state) that can be propagated downstream.

KL Regularization
-----------------
The KL term in the multi-task loss is the diagonal Gaussian KL:

    KL = 0.5 * mean_over_dims( sigma^2 + mu^2 - log(sigma^2) - 1 )

where sigma = g_phi(X) and mu = X (treating X as the variational mean).
This penalizes both large noise (sigma >> 1) and large state norms (|X| >> 1),
encouraging the diffusion term to be informative but not degenerate.

References
----------
    Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes."
        ICLR 2014.  KL divergence formula for diagonal Gaussian.
    Kidger et al. (2021). "Neural SDEs as Infinite-Dimensional GANs."
        ICML 2021.  Diffusion term design for latent SDEs.
    Li et al. (2020). "Scalable Gradients for SDEs." AISTATS 2020.
        beta-schedule annealing to prevent KL collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalDiffusionHead(nn.Module):
    """Per-dimension noise scale for the M1 Neural SDE diffusion term.

    Architecture:
        g_phi(X) = softplus( Linear(SiLU(Linear(X))) ) + noise_floor

    The noise_floor ensures g_phi > 0 at all times, preventing the
    diffusion term from collapsing (which would reduce the SDE to a CDE
    and stop all KL gradient signal).

    Parameters
    ----------
    hidden_dim : int
        Entity state dimension (= HeteroMemory.memory_dim).  Input and
        output dimension of g_phi.
    inner_dim : int | None
        Width of the hidden layer.  Defaults to 2 * hidden_dim.
    noise_floor : float
        Minimum per-dimension noise scale.  Default 1e-3 prevents exact
        zero output (KL collapse), while being small enough not to
        dominate the drift term early in training.
    """

    def __init__(
        self,
        hidden_dim: int,
        inner_dim: int | None = None,
        noise_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.noise_floor = noise_floor
        inner = inner_dim if inner_dim is not None else hidden_dim * 2
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_dim),
        )
        self._hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-dimension positive noise scale.

        Args:
            x: (..., hidden_dim) entity state tensor (any leading dims).

        Returns:
            Same shape as x — strictly positive noise scales g_phi(x).
        """
        return F.softplus(self.net(x)) + self.noise_floor

    def kl_divergence(
        self,
        z: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diagonal Gaussian KL divergence for the M1 variational objective.

        KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(sigma^2 + mu^2 - log(sigma^2) - 1)

        Args:
            z:     (..., hidden_dim) entity state (treated as variational mean mu).
            sigma: (..., hidden_dim) noise scales from forward(). If None,
                   computes forward(z) automatically.

        Returns:
            Scalar KL divergence (mean over all dimensions and batch).
        """
        if sigma is None:
            sigma = self.forward(z)
        sigma2 = sigma.pow(2)
        kl = 0.5 * (sigma2 + z.pow(2) - sigma2.log() - 1.0)
        return kl.mean()

    def sample_noise(
        self,
        z: torch.Tensor,
        dt: float,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Euler-Maruyama noise for the diffusion term.

        Computes g_phi(z) * sqrt(dt) * eps  where eps ~ N(0, I).
        During inference (training=False) returns zeros but still computes
        sigma for uncertainty estimation.

        Args:
            z:        (..., hidden_dim) current entity state.
            dt:       Euler-Maruyama step size in seconds.
            training: If False, returns zero noise (deterministic inference).

        Returns:
            (noise, sigma) where:
                noise: (..., hidden_dim) diffusion increment (zeros if not training).
                sigma: (..., hidden_dim) per-dimension noise scale g_phi(z).
        """
        sigma = self.forward(z)           # (..., hidden_dim)
        if training:
            eps = torch.randn_like(z)
            noise = sigma * eps * (dt ** 0.5)
        else:
            noise = torch.zeros_like(z)
        return noise, sigma
