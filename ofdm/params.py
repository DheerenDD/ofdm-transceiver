"""
System parameters for the OFDM transceiver.

Every number used anywhere in the project lives here, so experiments only
change a parameter and never hard-code values.

The defaults follow IEEE 802.11a/g (Wi-Fi) numerology:
20 MHz sampling, 64-point FFT, 0.8 us cyclic prefix.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class OFDMParams:
    # ---- Primary (chosen) parameters ----
    n_fft: int = 64                 # number of subcarriers = FFT size
    n_cp: int = 16                  # cyclic prefix length in samples
    fs: float = 20e6                # sample rate in Hz (= occupied bandwidth here)
    pilot_spacing: int = 4          # a pilot on every 4th subcarrier (comb type)
    mod_order: int = 4              # 4 = QPSK, 16 = 16-QAM
    n_symbols: int = 100            # OFDM symbols per simulated frame
    pilot_value: complex = 1 + 0j   # known pilot symbol (unit power)
    seed: int = 42                  # random seed so results are reproducible

    # ---- Default multipath channel (exponential power-delay profile) ----
    n_taps: int = 4                 # number of echo paths
    max_delay: int = 4              # delay of the last path in samples (must be <= n_cp)
    decay_db: float = 3.0           # each path is this many dB weaker than the previous

    # ---- Derived values (filled in automatically) ----
    pilot_idx: np.ndarray = field(init=False, repr=False)
    data_idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_cp >= self.n_fft:
            raise ValueError("Cyclic prefix must be shorter than the FFT size")
        if self.mod_order not in (4, 16):
            raise ValueError("mod_order must be 4 (QPSK) or 16 (16-QAM)")
        all_idx = np.arange(self.n_fft)
        self.pilot_idx = all_idx[:: self.pilot_spacing]
        self.data_idx = np.setdiff1d(all_idx, self.pilot_idx)

    # Subcarrier spacing: delta_f = fs / N
    @property
    def delta_f(self) -> float:
        return self.fs / self.n_fft

    # Useful symbol duration: T_u = 1 / delta_f = N / fs
    @property
    def t_useful(self) -> float:
        return self.n_fft / self.fs

    # Cyclic prefix duration: T_cp = N_cp / fs
    @property
    def t_cp(self) -> float:
        return self.n_cp / self.fs

    # Total OFDM symbol duration including the CP
    @property
    def t_symbol(self) -> float:
        return self.t_useful + self.t_cp

    # Fraction of time spent on the CP (time that carries no new data)
    @property
    def cp_overhead(self) -> float:
        return self.t_cp / self.t_symbol

    # Longest echo path difference the CP can absorb: c * T_cp
    @property
    def max_path_difference_m(self) -> float:
        return 3e8 * self.t_cp

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.mod_order))

    @property
    def n_data(self) -> int:
        return len(self.data_idx)

    @property
    def n_pilots(self) -> int:
        return len(self.pilot_idx)

    # Bits carried by one OFDM symbol (data subcarriers only)
    @property
    def bits_per_ofdm_symbol(self) -> int:
        return self.n_data * self.bits_per_symbol

    # Net data rate after pilot and CP overhead
    @property
    def data_rate(self) -> float:
        return self.bits_per_ofdm_symbol / self.t_symbol

    def summary(self) -> str:
        mod = {4: "QPSK", 16: "16-QAM"}[self.mod_order]
        lines = [
            "OFDM system parameters",
            "-" * 40,
            f"FFT size / subcarriers   : {self.n_fft}",
            f"Data / pilot subcarriers : {self.n_data} / {self.n_pilots}",
            f"Modulation               : {mod} ({self.bits_per_symbol} bit/symbol)",
            f"Sample rate              : {self.fs / 1e6:.1f} MHz",
            f"Subcarrier spacing       : {self.delta_f / 1e3:.1f} kHz",
            f"Useful symbol time       : {self.t_useful * 1e6:.2f} us",
            f"CP length                : {self.n_cp} samples = {self.t_cp * 1e6:.2f} us",
            f"Total symbol time        : {self.t_symbol * 1e6:.2f} us",
            f"CP overhead              : {self.cp_overhead * 100:.1f} %",
            f"Max echo path difference : {self.max_path_difference_m:.0f} m",
            f"Bits per OFDM symbol     : {self.bits_per_ofdm_symbol}",
            f"Net data rate            : {self.data_rate / 1e6:.1f} Mbit/s",
            f"Channel                  : {self.n_taps} taps, max delay {self.max_delay} samples"
            f" = {self.max_delay / self.fs * 1e6:.2f} us, {self.decay_db:.0f} dB/tap decay",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print(OFDMParams().summary())
    print()
    print(OFDMParams(mod_order=16).summary())