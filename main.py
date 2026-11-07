"""
Reproduce every result in this repository with one command:

    python main.py                 # checks + all experiments
    python main.py --only A C      # checks + selected experiments
    python main.py --list          # show what each experiment does

The sanity checks always run first. If any check fails, the experiments are
skipped so no plot is ever produced from a broken signal chain.
All figures are written to results/.
"""

import argparse
import sys
import time

from ofdm.params import OFDMParams
from tests import check_chain
from experiments import ber_vs_ebn0, link_visuals, cp_length, pilot_density

EXPERIMENTS = {
    "A": ("BER vs Eb/N0 (AWGN, Rayleigh, perfect vs LS CSI)", ber_vs_ebn0.main),
    "V": ("Link visuals (constellation, channel estimate)", link_visuals.main),
    "B": ("Cyclic prefix length vs delay spread", cp_length.main),
    "C": ("Pilot density trade-off", pilot_density.main),
}


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="OFDM transceiver: reproduce all results")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                        help="run only these experiments (default: all)")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args()

    if args.list:
        for key, (desc, _) in EXPERIMENTS.items():
            print(f"  {key}  {desc}")
        return 0

    selected = args.only or list(EXPERIMENTS)
    t_start = time.time()

    banner("System parameters")
    print(OFDMParams().summary())

    banner("Sanity checks")
    if not check_chain.main():
        print("\nChecks failed: fix the signal chain before running experiments.")
        return 1

    timings = {}
    for key in selected:
        desc, run = EXPERIMENTS[key]
        banner(f"Experiment {key}: {desc}")
        t0 = time.time()
        run()
        timings[key] = time.time() - t0

    banner("Summary")
    for key, t in timings.items():
        print(f"  {key}  {EXPERIMENTS[key][0]:<52s} {t:5.1f} s")
    print(f"\nTotal time {time.time() - t_start:.1f} s. Figures are in results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())