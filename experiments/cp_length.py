"""
Experiment B: what happens when the cyclic prefix is shorter than the channel?

A fixed multipath profile with a long delay spread (12 samples = 0.6 us) is
used, and the CP length is swept from 0 to 16 samples. The receiver is given
perfect channel knowledge, so any extra errors come only from inter-symbol
interference (ISI) and inter-carrier interference (ICI) caused by echoes that
spill past the CP.

Expected result:
    CP >= delay spread  -> BER follows Rayleigh theory
    CP <  delay spread  -> BER hits a floor that more SNR cannot remove
The price of a longer CP is lower throughput (right-hand plot).

Run from the project root:   python -m experiments.cp_length
Output:                       results/cp_length.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm.params import OFDMParams
from ofdm.link import simulate_ber, ber_theory_rayleigh

RESULTS = "results"
EBN0_DB = np.arange(0, 41, 4)
CP_LENGTHS = [0, 2, 4, 8, 12, 16]
CHANNEL = dict(n_taps=4, max_delay=12, decay_db=2.0)   # long, strong echoes
HIGH_SNR_DB = 40


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(11)
    floor_ber, rates = [], []

    print(f"Channel: {CHANNEL['n_taps']} taps, max delay {CHANNEL['max_delay']} samples")
    print(" CP | " + " | ".join(f"{e:>2d} dB  " for e in EBN0_DB))

    for n_cp in CP_LENGTHS:
        p = OFDMParams(n_cp=n_cp, n_symbols=10, **CHANNEL)
        bers = [simulate_ber(p, e, rng, "rayleigh", "perfect", min_frames=300) for e in EBN0_DB]
        style = "-o" if n_cp >= p.max_delay else "--s"
        ax1.semilogy(EBN0_DB, bers, style, ms=4,
                     label=f"CP = {n_cp:2d} samples ({p.t_cp * 1e6:.1f} us)")
        floor_ber.append(bers[-1])
        rates.append(p.data_rate / 1e6)
        print(f" {n_cp:2d} | " + " | ".join(f"{b:.1e}" for b in bers))

    e = np.linspace(EBN0_DB[0], EBN0_DB[-1], 300)
    ax1.semilogy(e, ber_theory_rayleigh(10 ** (e / 10), 4), "k:", lw=1.5,
                 label="Rayleigh theory (no ISI)")
    ax1.set_title(f"QPSK, delay spread {CHANNEL['max_delay']} samples, perfect CSI")
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit error rate")
    ax1.set_ylim(1e-5, 0.5)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    # Right: the trade-off at high SNR
    ax2.semilogy(CP_LENGTHS, floor_ber, "o-", color="tab:red", label=f"BER at {HIGH_SNR_DB} dB")
    ax2.axvline(CHANNEL["max_delay"], color="gray", ls="--")
    ax2.text(CHANNEL["max_delay"] + 0.3, max(floor_ber) * 0.5, "CP = delay spread",
             color="gray", fontsize=9)
    ax2.set_xlabel("CP length (samples)")
    ax2.set_ylabel("BER at high SNR", color="tab:red")
    ax2.grid(True, which="both", alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(CP_LENGTHS, rates, "s--", color="tab:blue", label="net data rate")
    ax2b.set_ylabel("Net data rate (Mbit/s)", color="tab:blue")
    ax2.set_title("Trade-off: robustness vs throughput")

    fig.tight_layout()
    path = os.path.join(RESULTS, "cp_length.png")
    fig.savefig(path, dpi=150)
    print("\n CP | BER @ 40 dB | rate (Mbit/s) | overhead")
    for n_cp, b, r in zip(CP_LENGTHS, floor_ber, rates):
        print(f" {n_cp:2d} | {b:11.1e} | {r:13.1f} | {n_cp / (64 + n_cp) * 100:5.1f} %")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()