"""
Experiment C: how many pilots are enough?

Pilots cost data subcarriers, but too few pilots cannot follow a
frequency-selective channel. Linear interpolation between pilots works only
if the channel changes slowly from one pilot to the next, i.e. if the pilot
spacing is small compared with the coherence bandwidth (roughly n_fft / delay
spread subcarriers).

Left:  BER vs Eb/N0 for pilot spacings 2, 4, 8, 16 (LS estimation)
Right: channel-estimation MSE without noise, vs pilot spacing, for three
       delay spreads (the interpolation error alone)

Run from the project root:   python -m experiments.pilot_density
Output:                       results/pilot_density.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm.params import OFDMParams
from ofdm.link import run_frame, simulate_ber, ber_theory_rayleigh

RESULTS = "results"
EBN0_DB = np.arange(0, 41, 4)
SPACINGS = [2, 4, 8, 16]
DELAYS = [2, 4, 8]
NOISELESS_DB = 200          # Eb/N0 high enough that noise is negligible


def interpolation_mse_db(spacing, max_delay, rng, n_frames=300):
    p = OFDMParams(pilot_spacing=spacing, max_delay=max_delay, n_symbols=1)
    mse = []
    for _ in range(n_frames):
        r = run_frame(p, NOISELESS_DB, rng, "rayleigh", "ls")
        mse.append(np.mean(np.abs(r["h_used"] - r["h_true"]) ** 2))
    return 10 * np.log10(np.mean(mse))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(21)

    # ---- Left: BER for each pilot spacing (default channel, 4-sample delay) ----
    print(" spacing | data SC | rate Mbit/s | " + " | ".join(f"{e:>2d} dB  " for e in EBN0_DB))
    for sp in SPACINGS:
        p = OFDMParams(pilot_spacing=sp, n_symbols=10)
        bers = [simulate_ber(p, e, rng, "rayleigh", "ls", min_frames=300) for e in EBN0_DB]
        ax1.semilogy(EBN0_DB, bers, "o-", ms=4,
                     label=f"every {sp:2d}: {p.n_pilots} pilots, {p.data_rate / 1e6:.0f} Mbit/s")
        print(f" {sp:7d} | {p.n_data:7d} | {p.data_rate / 1e6:11.1f} | "
              + " | ".join(f"{b:.1e}" for b in bers))

    p = OFDMParams(n_symbols=10)
    perfect = [simulate_ber(p, e, rng, "rayleigh", "perfect", min_frames=300) for e in EBN0_DB]
    ax1.semilogy(EBN0_DB, perfect, "k--", label="perfect CSI (reference)")
    e = np.linspace(EBN0_DB[0], EBN0_DB[-1], 300)
    ax1.semilogy(e, ber_theory_rayleigh(10 ** (e / 10), 4), "k:", lw=1, label="Rayleigh theory")
    ax1.set_title(f"QPSK, LS estimation, delay spread {p.max_delay} samples")
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit error rate")
    ax1.set_ylim(1e-5, 0.5)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    # ---- Right: interpolation error alone ----
    print("\nNoise-free interpolation MSE (dB)")
    print(" spacing | " + " | ".join(f"delay {d}" for d in DELAYS))
    table = np.zeros((len(SPACINGS), len(DELAYS)))
    for i, sp in enumerate(SPACINGS):
        for j, d in enumerate(DELAYS):
            table[i, j] = interpolation_mse_db(sp, d, rng)
        print(f" {sp:7d} | " + " | ".join(f"{v:7.1f}" for v in table[i]))
    for j, d in enumerate(DELAYS):
        ax2.plot(SPACINGS, table[:, j], "o-", label=f"delay spread {d} samples")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(SPACINGS)
    ax2.set_xticklabels([str(s) for s in SPACINGS])
    ax2.set_xlabel("Pilot spacing (subcarriers)")
    ax2.set_ylabel("Channel estimation MSE (dB)")
    ax2.set_title("Interpolation error without noise")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(RESULTS, "pilot_density.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()