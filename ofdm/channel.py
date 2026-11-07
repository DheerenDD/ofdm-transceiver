"""
Channel models: tapped-delay-line multipath (Rayleigh) and AWGN.

Multipath model
    h[n] = sum_k a_k * delta[n - d_k]
    a_k  ~ complex Gaussian (Rayleigh magnitude), average power from an
           exponential power-delay profile, normalised so E[sum |a_k|^2] = 1.
    Block fading: the taps stay constant over one frame and change per frame.

Noise scaling
    Each subcarrier carries Es = 1 and noise variance per subcarrier equals
    the noise variance per time sample (orthonormal FFT). So
        Es/N0 = bits_per_symbol * Eb/N0  ->  noise_var = 1 / (Es/N0)
    With include_overhead=True, the energy spent on the CP and on pilots is
    charged to the data bits as well, which shifts the BER curve to the right.
"""

import numpy as np
from .params import OFDMParams


def exponential_pdp(n_taps: int, max_delay: int, decay_db: float = 3.0):
    """Tap delays (samples) and average tap powers for an exponential profile.

    Taps are spread evenly from 0 to max_delay samples and each one is
    decay_db weaker than the previous. Powers are normalised to sum to 1.
    """
    if n_taps == 1:
        return np.array([0]), np.array([1.0])
    delays = np.round(np.linspace(0, max_delay, n_taps)).astype(int)
    powers = 10 ** (-decay_db * np.arange(n_taps) / 10)
    return delays, powers / powers.sum()


def rayleigh_taps(delays, powers, rng: np.random.Generator) -> np.ndarray:
    """Draw one random channel impulse response h[n]."""
    h = np.zeros(delays.max() + 1, dtype=complex)
    gains = (rng.standard_normal(len(delays)) + 1j * rng.standard_normal(len(delays))) / np.sqrt(2)
    h[delays] += gains * np.sqrt(powers)
    return h


def apply_multipath(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution with the channel, trimmed to the input length."""
    return np.convolve(signal, h)[: signal.size]


def frequency_response(h: np.ndarray, n_fft: int) -> np.ndarray:
    """True channel on each subcarrier: H[k] = FFT of h, zero-padded to n_fft."""
    return np.fft.fft(h, n_fft)


def ebn0_to_noise_var(ebn0_db: float, p: OFDMParams, include_overhead: bool = False) -> float:
    """Complex noise variance per sample for a given Eb/N0 in dB."""
    ebn0 = 10 ** (ebn0_db / 10)
    esn0 = p.bits_per_symbol * ebn0
    if include_overhead:
        cp_factor = (p.n_fft + p.n_cp) / p.n_fft     # CP energy carries no new bits
        pilot_factor = p.n_fft / p.n_data            # pilot energy carries no bits
        esn0 /= cp_factor * pilot_factor
    return 1.0 / esn0


def add_awgn(signal: np.ndarray, noise_var: float, rng: np.random.Generator) -> np.ndarray:
    """Add circular complex Gaussian noise with the given variance."""
    noise = (rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size))
    return signal + noise * np.sqrt(noise_var / 2)


if __name__ == "__main__":
    p = OFDMParams()
    rng = np.random.default_rng(p.seed)

    delays, powers = exponential_pdp(p.n_taps, p.max_delay, p.decay_db)
    print("Tap delays (samples):", delays, "| powers:", np.round(powers, 3))
    print(f"Max delay {delays.max()} samples = {delays.max() / p.fs * 1e6:.2f} us "
          f"(CP = {p.t_cp * 1e6:.2f} us)")

    # Average channel energy should be 1 over many realisations
    energies = [np.sum(np.abs(rayleigh_taps(delays, powers, rng)) ** 2) for _ in range(20_000)]
    print(f"Mean channel energy over 20k draws: {np.mean(energies):.3f}")

    # Noise check: measured variance should match the requested one
    nv = ebn0_to_noise_var(10, p)
    measured = np.var(add_awgn(np.zeros(200_000, complex), nv, rng))
    print(f"Noise var at Eb/N0=10 dB: requested {nv:.5f}, measured {measured:.5f}")
    nv_oh = ebn0_to_noise_var(10, p, include_overhead=True)
    print(f"With CP + pilot overhead charged: {nv_oh:.5f} "
          f"(penalty {10 * np.log10(nv_oh / nv):.2f} dB)")