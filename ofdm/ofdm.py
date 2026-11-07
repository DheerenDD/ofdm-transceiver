"""
OFDM modulator and demodulator.

Resource grid shape: (n_symbols, n_fft)
    rows    = OFDM symbols (time)
    columns = subcarriers  (frequency)

FFTs use norm="ortho", so the transform preserves energy: a grid with unit
power per subcarrier gives unit power per time-domain sample, and noise
added in time has the same variance per subcarrier after the receiver FFT.
"""

import numpy as np
from .params import OFDMParams


def build_grid(data_symbols: np.ndarray, p: OFDMParams) -> np.ndarray:
    """Place data symbols and known pilots onto the resource grid."""
    expected = p.n_symbols * p.n_data
    if data_symbols.size != expected:
        raise ValueError(f"Need {expected} data symbols, got {data_symbols.size}")
    grid = np.zeros((p.n_symbols, p.n_fft), dtype=complex)
    grid[:, p.data_idx] = data_symbols.reshape(p.n_symbols, p.n_data)
    grid[:, p.pilot_idx] = p.pilot_value
    return grid


def extract_data(grid: np.ndarray, p: OFDMParams) -> np.ndarray:
    """Pull the data subcarriers out of a grid as a flat symbol array."""
    return grid[:, p.data_idx].ravel()


def modulate(grid: np.ndarray, p: OFDMParams) -> np.ndarray:
    """Grid -> serial time-domain signal (IFFT per symbol, then add CP)."""
    time_symbols = np.fft.ifft(grid, axis=1, norm="ortho")
    # Copy the last n_cp samples. Index from n_fft - n_cp rather than -n_cp,
    # because x[:, -0:] would return the whole symbol when n_cp = 0.
    cp = time_symbols[:, p.n_fft - p.n_cp:]
    with_cp = np.hstack([cp, time_symbols])        # ... to the front
    return with_cp.ravel()                         # parallel -> serial


def demodulate(signal: np.ndarray, p: OFDMParams) -> np.ndarray:
    """Serial time-domain signal -> grid (remove CP, FFT per symbol)."""
    sym_len = p.n_fft + p.n_cp
    n_sym = signal.size // sym_len
    blocks = signal[: n_sym * sym_len].reshape(n_sym, sym_len)   # serial -> parallel
    no_cp = blocks[:, p.n_cp:]                                   # drop the CP
    return np.fft.fft(no_cp, axis=1, norm="ortho")


if __name__ == "__main__":
    from .modulation import random_bits, map_bits, demap_symbols

    p = OFDMParams()
    rng = np.random.default_rng(p.seed)
    bits = random_bits(p.n_symbols * p.bits_per_ofdm_symbol, rng)
    grid = build_grid(map_bits(bits, p.mod_order), p)
    tx = modulate(grid, p)
    rx_grid = demodulate(tx, p)

    sym_len = p.n_fft + p.n_cp
    first = tx[:sym_len]
    print(f"TX samples          : {tx.size} (= {p.n_symbols} x {sym_len})")
    print(f"Mean sample power   : {np.mean(np.abs(tx) ** 2):.3f}")
    print(f"CP equals symbol end: {np.allclose(first[:p.n_cp], first[-p.n_cp:])}")
    print(f"Max grid error      : {np.max(np.abs(rx_grid - grid)):.2e}")
    back = demap_symbols(extract_data(rx_grid, p), p.mod_order)
    print(f"Ideal-channel bit errors: {np.count_nonzero(back != bits)}")