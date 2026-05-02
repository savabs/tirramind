"""Spectral decomposition — FFT power spectrum and CWT scalogram."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def power_spectrum(data: np.ndarray, sampling_freq: float = 52.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided power spectrum via FFT.

    Parameters
    ----------
    data : 1-D array.
    sampling_freq : Samples per year (52 for weekly data).

    Returns
    -------
    (frequencies, power) — frequencies in cycles/year, power in |X(f)|^2.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    N = len(data)

    # Remove mean to avoid DC spike dominating
    centered = data - data.mean()

    # FFT
    fft_vals = rfft(centered)
    freqs = rfftfreq(N, d=1.0 / sampling_freq)  # cycles per year
    power = np.abs(fft_vals) ** 2 / N

    return freqs, power


def scalogram(
    data: np.ndarray,
    periods: np.ndarray | None = None,
    sampling_freq: float = 52.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous wavelet transform scalogram using Morlet wavelet.

    Parameters
    ----------
    data : 1-D array of length T.
    periods : 1-D array of periods to analyze (in years). If None,
              uses logarithmically spaced periods from 4 weeks to 4 years.
    sampling_freq : Samples per year (52 for weekly).

    Returns
    -------
    (times, periods, power_matrix) — times is np.arange(T),
    periods in years, power_matrix shape (len(periods), T).
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    T = len(data)

    if periods is None:
        # Default: 4 weeks (~0.08 yr) to 4 years, log-spaced
        periods = np.geomspace(4 / sampling_freq, 4.0, num=50)

    # Convert periods (years) to widths (samples) for Morlet wavelet
    w = 6.0  # standard Morlet parameter
    widths = periods * sampling_freq * w / (2 * np.pi)

    # Compute CWT via convolution with scaled Morlet wavelets
    centered = data - data.mean()
    coeffs = _cwt_morlet(centered, widths, w)
    power_matrix = np.abs(coeffs) ** 2

    times = np.arange(T)
    return times, periods, power_matrix


def _cwt_morlet(data: np.ndarray, widths: np.ndarray, w: float = 6.0) -> np.ndarray:
    """Continuous wavelet transform using Morlet wavelet (pure numpy)."""
    T = len(data)
    n_scales = len(widths)
    output = np.zeros((n_scales, T), dtype=np.complex128)

    for i, width in enumerate(widths):
        # Generate Morlet wavelet at this scale
        half = int(4 * width + 0.5)
        t = np.arange(-half, half + 1)
        wavelet = np.exp(1j * w * t / width) * np.exp(-0.5 * (t / width) ** 2) / np.sqrt(width)
        # Full convolution, then center-crop to match data length
        conv = np.convolve(data, wavelet, mode="full")
        start = (len(conv) - T) // 2
        output[i] = conv[start : start + T]

    return output
