"""Zernike function tests."""

import numpy as np
from psfsim import zernike


def test_mapping():
    """Noll mapping test"""

    sol = [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0)]
    for j in range(len(sol)):
        n, m = zernike.noll_to_zernike(j + 1)
        print(n, m)
        assert sol[j] == (n, m)


def test_zernike():
    """Zernike function test."""

    x, y = np.meshgrid(np.linspace(-0.7, 0.7, 15), np.linspace(-0.7, 0.7, 15))
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Z1
    target = 1.0
    assert np.all(np.abs(target - zernike.zernike(0, 0, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(target - zernike.zernike(0, 0, rho, theta, normalized=True)) < 1e-6)

    # Z2
    target = x
    assert np.all(np.abs(target - zernike.zernike(1, 1, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(2.0 * target - zernike.zernike(1, 1, rho, theta, normalized=True)) < 1e-6)

    # Z3
    target = y
    assert np.all(np.abs(target - zernike.zernike(1, -1, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(2.0 * target - zernike.zernike(1, -1, rho, theta, normalized=True)) < 1e-6)

    # Z4
    target = (x**2 + y**2 - 0.5) * 2.0
    assert np.all(np.abs(target - zernike.zernike(2, 0, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(3.0) * target - zernike.zernike(2, 0, rho, theta, normalized=True)) < 1e-6)

    # Z5
    target = 2.0 * x * y
    assert np.all(np.abs(target - zernike.zernike(2, -2, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(6.0) * target - zernike.zernike(2, -2, rho, theta, normalized=True)) < 1e-6)

    # Z6
    target = x**2 - y**2
    assert np.all(np.abs(target - zernike.zernike(2, 2, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(6.0) * target - zernike.zernike(2, 2, rho, theta, normalized=True)) < 1e-6)

    # Z7
    target = (3.0 * (x**2 + y**2) - 2.0) * y
    assert np.all(np.abs(target - zernike.zernike(3, -1, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(8.0) * target - zernike.zernike(3, -1, rho, theta, normalized=True)) < 1e-6)

    # Z8
    target = (3.0 * (x**2 + y**2) - 2.0) * x
    assert np.all(np.abs(target - zernike.zernike(3, 1, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(8.0) * target - zernike.zernike(3, 1, rho, theta, normalized=True)) < 1e-6)

    # Z9
    target = 3.0 * x**2 * y - y**3
    assert np.all(np.abs(target - zernike.zernike(3, -3, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(8.0) * target - zernike.zernike(3, -3, rho, theta, normalized=True)) < 1e-6)

    # Z10
    target = x**3 - 3.0 * x * y**2
    assert np.all(np.abs(target - zernike.zernike(3, 3, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(8.0) * target - zernike.zernike(3, 3, rho, theta, normalized=True)) < 1e-6)

    # Z11
    target = 6.0 * (x**2 + y**2) ** 2 - 6.0 * (x**2 + y**2) + 1.0
    assert np.all(np.abs(target - zernike.zernike(4, 0, rho, theta, normalized=False)) < 1e-6)
    assert np.all(np.abs(np.sqrt(5.0) * target - zernike.zernike(4, 0, rho, theta, normalized=True)) < 1e-6)
