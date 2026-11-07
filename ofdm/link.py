"""
End-to-end link simulation: bits -> OFDM -> channel -> receiver -> BER.

Shared by all experiments so each experiment only chooses parameters.
"""

import numpy as np
from scipy.special import erfc

from .params import OFDMParams
from .modulation import random_bits, map_bits, demap_symbols
from .ofdm import build_grid, extract_data, modulate, demodulate
from .channel import (exponential_pdp, rayleigh_taps, apply_multipath,
                      frequency_response, ebn0_to_noise_var, add_awgn)
from .receiver import ls_estimate, equalize_zf


def run_frame(p: OFDMParams, ebn0_db: float, rng: np.random.Generator,
              channel: str = "rayleigh", csi: str = "ls"):
    """Simulate one frame. Returns a dict with bits, grids and channel info.

    channel: "awgn" (no multipath) or "rayleigh" (tapped delay line)
    csi:     "perfect" (receiver knows H) or "ls" (estimated from pilots)
    """
    bits = random_bits(p.n_symbols * p.bits_per_ofdm_symbol, rng)
    tx_grid = build_grid(map_bits(bits, p.mod_order), p)
    tx = modulate(tx_grid, p)

    if channel == "awgn":
        h = np.array([1.0 + 0j])
    elif channel == "rayleigh":
        delays, powers = exponential_pdp(p.n_taps, p.max_delay, p.decay_db)
        h = rayleigh_taps(delays, powers, rng)
    else:
        raise ValueError("channel must be 'awgn' or 'rayleigh'")

    noise_var = ebn0_to_noise_var(ebn0_db, p)
    rx = add_awgn(apply_multipath(tx, h), noise_var, rng)
    rx_grid = demodulate(rx, p)

    h_true = np.broadcast_to(frequency_response(h, p.n_fft), rx_grid.shape)
    if csi == "perfect":
        h_used = h_true
    elif csi == "ls":
        h_used = ls_estimate(rx_grid, p)
    else:
        raise ValueError("csi must be 'perfect' or 'ls'")

    eq_grid = equalize_zf(rx_grid, h_used)
    bits_hat = demap_symbols(extract_data(eq_grid, p), p.mod_order)

    return {
        "bits": bits, "bits_hat": bits_hat,
        "rx_grid": rx_grid, "eq_grid": eq_grid,
        "h_true": h_true, "h_used": h_used, "noise_var": noise_var,
    }


def simulate_ber(p: OFDMParams, ebn0_db: float, rng: np.random.Generator,
                 channel: str = "rayleigh", csi: str = "ls",
                 min_errors: int = 300, min_frames: int = 500,
                 max_bits: int = 5_000_000) -> float:
    """Monte Carlo BER.

    Keeps running frames until BOTH enough errors are counted and enough
    frames are run, or the bit budget is spent. The frame minimum matters for
    fading: every frame has one random channel, and a single deep fade can
    produce hundreds of errors, so the BER must be averaged over many
    independent channel realisations, not just many bits.
    """
    errors = total = frames = 0
    while (errors < min_errors or frames < min_frames) and total < max_bits:
        r = run_frame(p, ebn0_db, rng, channel, csi)
        errors += np.count_nonzero(r["bits"] != r["bits_hat"])
        total += r["bits"].size
        frames += 1
    return errors / total


# ---- Theoretical BER (Gray coding, Eb/N0 in linear units) ----

def ber_theory_awgn(ebn0: np.ndarray, mod_order: int) -> np.ndarray:
    if mod_order == 4:
        return 0.5 * erfc(np.sqrt(ebn0))                    # exact
    if mod_order == 16:
        return 0.375 * erfc(np.sqrt(0.4 * ebn0))            # nearest-neighbour approx.
    raise ValueError


def ber_theory_rayleigh(ebn0: np.ndarray, mod_order: int) -> np.ndarray:
    """Average of the AWGN BER over Rayleigh fading:
    E[Q(sqrt(2 a g))] = 0.5 * (1 - sqrt(a g / (1 + a g)))"""
    if mod_order == 4:
        return 0.5 * (1 - np.sqrt(ebn0 / (1 + ebn0)))       # exact
    if mod_order == 16:
        g = 0.4 * ebn0
        return 0.375 * (1 - np.sqrt(g / (1 + g)))           # approx.
    raise ValueError