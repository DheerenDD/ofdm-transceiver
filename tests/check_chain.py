"""
End-to-end sanity checks for the OFDM chain. Run from the project root:

    python -m tests.check_chain

Each check prints PASS or FAIL. They should all pass before any experiment.
"""

import numpy as np
from scipy.special import erfc

from ofdm.params import OFDMParams
from ofdm.modulation import random_bits, map_bits, demap_symbols
from ofdm.ofdm import build_grid, extract_data, modulate, demodulate
from ofdm.channel import (exponential_pdp, rayleigh_taps, apply_multipath,
                          frequency_response, ebn0_to_noise_var, add_awgn)
from ofdm.receiver import ls_estimate, equalize_zf, estimation_mse


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def transmit(p, rng):
    bits = random_bits(p.n_symbols * p.bits_per_ofdm_symbol, rng)
    grid = build_grid(map_bits(bits, p.mod_order), p)
    return bits, grid, modulate(grid, p)


def main():
    results = []

    # 1. Modulation round trip (both orders)
    rng = np.random.default_rng(1)
    for M in (4, 16):
        b = random_bits(40_000, rng)
        s = map_bits(b, M)
        errs = np.count_nonzero(demap_symbols(s, M) != b)
        es = np.mean(np.abs(s) ** 2)
        results.append(report(f"{M}-QAM round trip", errs == 0 and abs(es - 1) < 0.02,
                              f"errors={errs}, Es={es:.3f}"))

    # 2. Ideal channel: OFDM mod -> demod is lossless
    p = OFDMParams()
    bits, grid, tx = transmit(p, rng)
    rx_grid = demodulate(tx, p)
    errs = np.count_nonzero(demap_symbols(extract_data(rx_grid, p), p.mod_order) != bits)
    results.append(report("Ideal channel, zero bit errors", errs == 0, f"errors={errs}"))

    # 3. Multipath, no noise, delay < CP: LS + ZF recovers everything
    delays, powers = exponential_pdp(p.n_taps, p.max_delay, p.decay_db)
    h = rayleigh_taps(delays, powers, rng)
    rx_grid = demodulate(apply_multipath(tx, h), p)
    h_true = frequency_response(h, p.n_fft)
    # Because the CP is longer than the channel, each received subcarrier is
    # exactly the transmitted one times a single complex gain H[k]
    exact = np.allclose(rx_grid, grid * h_true[None, :])
    results.append(report("Y = H.X per subcarrier (CP works)", exact))
    x_hat = equalize_zf(rx_grid, h_true[None, :])
    errs = np.count_nonzero(demap_symbols(extract_data(x_hat, p), p.mod_order) != bits)
    results.append(report("Multipath + perfect CSI, zero errors", errs == 0, f"errors={errs}"))

    # 4. LS estimate is close to the true channel without noise
    h_est = ls_estimate(rx_grid, p)
    mse_pilots = estimation_mse(h_est[:, p.pilot_idx], h_true[None, p.pilot_idx])
    mse_all = estimation_mse(h_est, h_true[None, :])
    results.append(report("LS exact at pilots (no noise)", mse_pilots < 1e-20,
                          f"MSE={mse_pilots:.1e}"))
    results.append(report("LS interpolation MSE below -15 dB (no noise)",
                          10 * np.log10(mse_all) < -15,
                          f"MSE={mse_all:.1e} ({10 * np.log10(mse_all):.1f} dB)"))

    # 5. AWGN BER matches theory for QPSK and 16-QAM
    for M, ebn0_db in ((4, 6.0), (16, 8.0)):
        p = OFDMParams(mod_order=M, n_symbols=4000)
        bits, grid, tx = transmit(p, rng)
        nv = ebn0_to_noise_var(ebn0_db, p)
        rx_grid = demodulate(add_awgn(tx, nv, rng), p)
        ber = np.mean(demap_symbols(extract_data(rx_grid, p), M) != bits)
        g = 10 ** (ebn0_db / 10)
        theory = 0.5 * erfc(np.sqrt(g)) if M == 4 else 0.375 * erfc(np.sqrt(0.4 * g))
        ok = abs(ber - theory) / theory < 0.1
        results.append(report(f"{M}-QAM AWGN BER at {ebn0_db} dB", ok,
                              f"sim={ber:.2e}, theory={theory:.2e}"))

    print()
    print(f"{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()