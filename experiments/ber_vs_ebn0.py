"""
Experiment A: BER vs Eb/N0

For QPSK and 16-QAM, compares
    - AWGN channel                      (simulation vs theory)
    - Rayleigh multipath, perfect CSI   (simulation vs theory)
    - Rayleigh multipath, LS estimate   (simulation only)

Run from the project root:   python -m experiments.ber_vs_ebn0
Output:                       results/ber_vs_ebn0.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm.params import OFDMParams
from ofdm.link import simulate_ber, ber_theory_awgn, ber_theory_rayleigh

RESULTS = "results"
EBN0_DB = np.arange(0, 31, 2)
BER_FLOOR = 1e-5          # stop a curve once it drops below this


def sweep(p, channel, csi, rng):
    out = []
    for e in EBN0_DB:
        ber = simulate_ber(p, e, rng, channel, csi)
        out.append(ber)
        if ber < BER_FLOOR:
            break
    return np.array(out)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ebn0_fine = np.linspace(EBN0_DB[0], EBN0_DB[-1], 300)
    lin = 10 ** (ebn0_fine / 10)

    for ax, M in zip(axes, (4, 16)):
        # Short frames (10 symbols) = more independent channel draws per bit
        p = OFDMParams(mod_order=M, n_symbols=10)
        rng = np.random.default_rng(p.seed)
        name = {4: "QPSK", 16: "16-QAM"}[M]
        t0 = time.time()

        awgn = sweep(p, "awgn", "perfect", rng)
        ray_perfect = sweep(p, "rayleigh", "perfect", rng)
        ray_ls = sweep(p, "rayleigh", "ls", rng)

        ax.semilogy(ebn0_fine, ber_theory_awgn(lin, M), "k-", lw=1, label="AWGN theory")
        ax.semilogy(EBN0_DB[:awgn.size], awgn, "ko", ms=5, label="AWGN sim")
        ax.semilogy(ebn0_fine, ber_theory_rayleigh(lin, M), "b-", lw=1, label="Rayleigh theory")
        ax.semilogy(EBN0_DB[:ray_perfect.size], ray_perfect, "bs", ms=5, label="Rayleigh sim, perfect CSI")
        ax.semilogy(EBN0_DB[:ray_ls.size], ray_ls, "r^--", ms=5, label="Rayleigh sim, LS estimate")

        ax.set_title(f"{name} over OFDM ({p.n_fft} subcarriers, CP {p.n_cp})")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylim(1e-5, 0.5)
        ax.set_xlim(EBN0_DB[0], EBN0_DB[-1])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

        print(f"\n{name}  ({time.time() - t0:.0f} s)")
        print(" Eb/N0 |   AWGN     | Ray perfect | Ray LS")
        for i, e in enumerate(EBN0_DB):
            cols = [arr[i] if i < arr.size else np.nan for arr in (awgn, ray_perfect, ray_ls)]
            print(f" {e:4d}  | " + " | ".join(f"{c:10.2e}" for c in cols))

    axes[0].set_ylabel("Bit error rate")
    fig.tight_layout()
    path = os.path.join(RESULTS, "ber_vs_ebn0.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()