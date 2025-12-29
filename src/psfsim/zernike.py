# With assistance from DeepSeek
import numpy as np
from scipy.special import jacobi


def zernike_radial(n, m, rho):
    """
    Cross checked with https://desc-docs.readthedocs.io/en/v0.12.0/notebooks/zernike_eval.html
    MAY HAVE A MINUS SIGN ERROR
    Compute radial part of Zernike polynomial Rnm(rho)
    Args:
        n: azimuthal order (n >= 0)
        m: radial order (abs m <= n, n-m even)
        rho: radial coordinate array (0 <= rho <= 1)
    Returns:
        Radial polynomial values
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    k = (n - m) // 2
    alpha = m
    beta = 0
    poly = jacobi(k, alpha, beta)(2 * rho**2 - 1)
    return (-1) ** k * rho**m * poly


def zernike(n, m, rho, theta, normalized=True):
    """
    Complete Zernike polynomial Z_n^m(rho, theta)
    Args:
        n: azimuthal order
        m: radial order
        rho: radial coordinates (0 <= rho <= 1)
        theta: angular coordinates
        normalized: whether to normalize
    Returns:
        Complex array if m != 0, real array otherwise
    """
    if abs(m) > n:
        raise ValueError("Require abs m <= n")

    R = zernike_radial(n, abs(m), rho)
    norm = np.sqrt(2 * (n + 1) / (1 + (m == 0))) if normalized else 1.0

    if m > 0:
        return norm * R * np.cos(m * theta)
    elif m < 0:
        return norm * R * np.sin(-m * theta)
    else:
        return norm * R


# Precompute Noll indices if needed
def noll_to_zernike(j):
    """
    Taken from https://louisdesdoigts.github.io/dLux/API/utils/zernikes/
    Calculate the radial and azimuthal orders of the Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    n, m : tuple[int]
        The radial and azimuthal orders of the Zernike polynomial.
    """
    n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
    smallest_j_in_row = n * (n + 1) / 2 + 1
    number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
    sign_of_shift = -(j & 1) + ~(j & 1) + 2
    base_case = n & 1
    m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
    return int(n), int(m)
