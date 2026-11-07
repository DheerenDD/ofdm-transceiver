"""
Gray-coded QPSK and 16-QAM mapping and hard-decision demapping.

Constellations are normalised to unit average symbol energy (Es = 1), so
SNR and noise levels mean the same thing for both modulation orders.

Gray coding: neighbouring constellation points differ in exactly one bit,
so the most likely symbol error (to a neighbour) causes only one bit error.
"""

import numpy as np

# One axis of 16-QAM carries 2 bits. Gray order along the axis:
#   bits 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
_PAM4_LEVELS = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_PAM4_LUT = np.zeros((2, 2))
for (b0, b1), level in _PAM4_LEVELS.items():
    _PAM4_LUT[b0, b1] = level

QPSK_SCALE = 1 / np.sqrt(2)    # points at (+-1 +-1j)/sqrt(2)  -> Es = 1
QAM16_SCALE = 1 / np.sqrt(10)  # mean of |(+-1,+-3) +-j(+-1,+-3)|^2 is 10


def random_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Equally likely random bits as a uint8 array."""
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def map_bits(bits: np.ndarray, mod_order: int) -> np.ndarray:
    """Map a bit array onto complex constellation symbols."""
    bits = np.asarray(bits, dtype=np.uint8)
    k = int(np.log2(mod_order))
    if bits.size % k:
        raise ValueError(f"Number of bits must be a multiple of {k}")
    b = bits.reshape(-1, k)

    if mod_order == 4:
        # bit 0 -> I sign, bit 1 -> Q sign (0 -> +1, 1 -> -1)
        i = 1 - 2 * b[:, 0].astype(float)
        q = 1 - 2 * b[:, 1].astype(float)
        return (i + 1j * q) * QPSK_SCALE

    if mod_order == 16:
        # bits 0,1 -> I level, bits 2,3 -> Q level
        i = _PAM4_LUT[b[:, 0], b[:, 1]]
        q = _PAM4_LUT[b[:, 2], b[:, 3]]
        return (i + 1j * q) * QAM16_SCALE

    raise ValueError("mod_order must be 4 or 16")


def _pam4_decide(x: np.ndarray) -> np.ndarray:
    """Hard decision on one 16-QAM axis (already de-normalised). Returns 2 bits."""
    b0 = (x > 0).astype(np.uint8)            # sign decides the first bit
    b1 = (np.abs(x) < 2).astype(np.uint8)    # inner levels (+-1) have b1 = 1
    return np.stack([b0, b1], axis=1)


def demap_symbols(symbols: np.ndarray, mod_order: int) -> np.ndarray:
    """Hard-decision demapping: nearest constellation point back to bits."""
    s = np.asarray(symbols).ravel()

    if mod_order == 4:
        b0 = (s.real < 0).astype(np.uint8)
        b1 = (s.imag < 0).astype(np.uint8)
        return np.stack([b0, b1], axis=1).ravel()

    if mod_order == 16:
        s = s / QAM16_SCALE
        bi = _pam4_decide(s.real)
        bq = _pam4_decide(s.imag)
        return np.hstack([bi, bq]).ravel()

    raise ValueError("mod_order must be 4 or 16")


def constellation(mod_order: int) -> np.ndarray:
    """All ideal constellation points (useful for plotting)."""
    k = int(np.log2(mod_order))
    all_bits = ((np.arange(mod_order)[:, None] >> np.arange(k)[::-1]) & 1)
    return map_bits(all_bits.ravel().astype(np.uint8), mod_order)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for M in (4, 16):
        bits = random_bits(12_000, rng)
        syms = map_bits(bits, M)
        back = demap_symbols(syms, M)
        es = np.mean(np.abs(syms) ** 2)
        errors = np.count_nonzero(bits != back)
        print(f"M={M:2d}: {syms.size} symbols, Es = {es:.3f}, round-trip bit errors = {errors}")