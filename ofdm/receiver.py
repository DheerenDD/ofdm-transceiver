"""
Receiver: pilot-based least-squares channel estimation and one-tap equalisation.

Because the CP turns the channel into a circular convolution, every
subcarrier sees a single complex gain:   Y[k] = H[k] * X[k] + W[k]
So equalisation is one complex division per subcarrier ("one-tap").
"""

import numpy as np
from .params import OFDMParams


def ls_estimate(rx_grid: np.ndarray, p: OFDMParams) -> np.ndarray:
    """Least-squares channel estimate on every subcarrier of every symbol.

    1. At pilot subcarriers:  H_ls = Y / X_pilot
    2. Between pilots: linear interpolation (real and imaginary parts)
    3. After the last pilot: linear extrapolation from the last two pilots
    """
    pilots_rx = rx_grid[:, p.pilot_idx]
    h_pilots = pilots_rx / p.pilot_value                  # shape (n_symbols, n_pilots)

    k_all = np.arange(p.n_fft)
    k_p = p.pilot_idx
    h_est = np.empty((rx_grid.shape[0], p.n_fft), dtype=complex)
    slope = (h_pilots[:, -1] - h_pilots[:, -2]) / (k_p[-1] - k_p[-2])

    for s in range(rx_grid.shape[0]):
        re = np.interp(k_all, k_p, h_pilots[s].real)
        im = np.interp(k_all, k_p, h_pilots[s].imag)
        h_est[s] = re + 1j * im

    # np.interp holds the edge value constant; replace with a linear extrapolation
    tail = k_all > k_p[-1]
    h_est[:, tail] = h_pilots[:, -1:] + slope[:, None] * (k_all[tail] - k_p[-1])
    return h_est


def equalize_zf(rx_grid: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-forcing: X_hat = Y / H. Fully undoes the channel but boosts noise
    on subcarriers in a deep fade (where |H| is small)."""
    return rx_grid / h


def equalize_mmse(rx_grid: np.ndarray, h: np.ndarray, noise_var: float) -> np.ndarray:
    """MMSE: X_hat = conj(H) Y / (|H|^2 + noise_var).

    Limits noise enhancement in fades by shrinking the output towards zero.
    That shrinkage biases the amplitude, which would break 16-QAM decision
    thresholds, so the bias is removed afterwards. For a single tap the
    unbiased MMSE output is then the same as ZF for hard decisions; MMSE
    makes a real difference for soft decisions (LLRs) and for MIMO, where
    streams interfere with each other.
    """
    g = np.conj(h) / (np.abs(h) ** 2 + noise_var)
    x_hat = g * rx_grid
    bias = (np.abs(h) ** 2) / (np.abs(h) ** 2 + noise_var)
    return x_hat / bias


def estimation_mse(h_est: np.ndarray, h_true: np.ndarray) -> float:
    """Mean squared error of the channel estimate."""
    return float(np.mean(np.abs(h_est - h_true) ** 2))