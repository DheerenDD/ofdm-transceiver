"""
Link visuals for one Rayleigh frame (16-QAM, Eb/N0 = 20 dB):
    1. Received constellation before vs after equalisation
    2. Channel frequency response: true vs LS estimate (pilots marked)

Run from the project root:   python -m experiments.link_visuals
Output:                       results/constellation.png, results/channel_estimate.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm.params import OFDMParams
from ofdm.link import run_frame
from ofdm.modulation import constellation

RESULTS = "results"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    p = OFDMParams(mod_order=16, n_symbols=20)
    rng = np.random.default_rng(7)
    r = run_frame(p, ebn0_db=20, rng=rng, channel="rayleigh", csi="ls")
    ideal = constellation(16)

    # ---- 1. Constellations ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    before = r["rx_grid"][:, p.data_idx].ravel()
    after = r["eq_grid"][:, p.data_idx].ravel()
    for ax, pts, title in ((axes[0], before, "Before equalisation (Y = H.X + W)"),
                           (axes[1], after, "After LS + one-tap ZF equalisation")):
        ax.plot(pts.real, pts.imag, ".", ms=3, alpha=0.5, label="received")
        ax.plot(ideal.real, ideal.imag, "r+", ms=12, mew=2, label="ideal 16-QAM")
        ax.set_title(title)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        lim = 1.6 if pts is after else max(1.6, np.percentile(np.abs(pts), 99) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    axes[1].legend(loc="upper right", fontsize=8)
    ber = np.mean(r["bits"] != r["bits_hat"])
    fig.suptitle(f"16-QAM, Rayleigh multipath, Eb/N0 = 20 dB, frame BER = {ber:.1e}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "constellation.png"), dpi=150)

    # ---- 2. Channel estimate (first OFDM symbol) ----
    k = np.arange(p.n_fft)
    h_true = r["h_true"][0]
    h_est = r["h_used"][0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(k, 20 * np.log10(np.abs(h_true)), "k-", lw=2, label="true |H|")
    axes[0].plot(k, 20 * np.log10(np.abs(h_est)), "r--", label="LS estimate (interpolated)")
    axes[0].plot(p.pilot_idx, 20 * np.log10(np.abs(h_est[p.pilot_idx])), "ro", label="pilots")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].legend(fontsize=8)
    axes[1].plot(k, np.unwrap(np.angle(h_true)), "k-", lw=2, label="true phase")
    axes[1].plot(k, np.unwrap(np.angle(h_est)), "r--", label="LS estimate")
    axes[1].plot(p.pilot_idx, np.unwrap(np.angle(h_est))[p.pilot_idx], "ro")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xlabel("Subcarrier index")
    axes[1].axvspan(p.pilot_idx[-1], p.n_fft - 1, color="orange", alpha=0.15,
                    label="extrapolated (after last pilot)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Frequency-selective channel: true response vs pilot-based LS estimate")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "channel_estimate.png"), dpi=150)

    print(f"Frame BER: {ber:.2e}")
    print("Saved results/constellation.png and results/channel_estimate.png")


if __name__ == "__main__":
    main()