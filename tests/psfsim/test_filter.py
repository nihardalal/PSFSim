"""Test functions for the filter."""

import numpy as np
from psfsim import filter_detector_properties as fdp


def test_indices():
    """Regression test for index of refraction."""

    n = [fdp.n_mercadtel(0.5 + 0.5 * j) for j in range(5)]
    ntarget = [
        (3.535003813489665 + 1.397975504523603j),
        (3.290866600993655 + 0.41166768695546824j),
        (3.2914586834935 + 0.2761795795767501j),
        (3.3389167117314136 + 0.17603053538895572j),
        (3.374851000897359 + 0.05450239626837025j),
    ]

    print(n)
    for j in range(5):
        assert np.abs(n[j] - ntarget[j]) < 1e-3


def test_rotation():
    """Test of rotation matrices."""

    # Rotation of longitude between FPA (x,y,z) and plane of incidence coordinates.
    s_ = -0.2 * np.arange(-2, 3)
    ux, uy = np.meshgrid(s_, s_)
    RT = fdp.local_to_fpa_rotation(ux, uy, 1)
    assert np.shape(RT) == (5, 5, 3, 3)
    assert np.all(
        np.abs(
            RT[0, 1, :, :]
            - np.array([[0.89442719, 0.4472136, 0.0], [-0.4472136, 0.89442719, 0.0], [0.0, 0.0, 1.0]])
        )
        < 1e-5
    )
    RT = fdp.local_to_fpa_rotation(ux, uy, -1)
    assert np.all(
        np.abs(
            RT[0, 1, :, :]
            - np.array([[-0.89442719, 0.4472136, 0.0], [0.4472136, 0.89442719, 0.0], [0.0, 0.0, -1.0]])
        )
        < 1e-5
    )

    # build an electric field
    Ex = np.zeros((5, 5), dtype=np.complex128)
    Ey = np.zeros((5, 5), dtype=np.complex128)
    Ez = np.zeros((5, 5), dtype=np.complex128)
    # make sure it's transverse
    w = np.sqrt(1 - ux**2 - uy**2)  # 3rd axis, uz
    Ex[:, :] += np.sqrt(1.0 - ux**2)
    Ey[:, :] -= ux * uy / np.sqrt(1.0 - ux**2)
    Ez[:, :] -= ux * w / np.sqrt(1.0 - ux**2)
    # ... and circularly polarized
    Ex[:, :] += 1j * (uy * Ez.real - w * Ey.real)
    Ey[:, :] += 1j * (w * Ex.real - ux * Ez.real)
    Ez[:, :] += 1j * (ux * Ey.real - uy * Ex.real)

    print(Ex[0, 1], Ey[0, 1], Ez[0, 1])
    assert np.all(np.abs(ux * Ex + uy * Ey + w * Ez) < 1e-10)

    # Now convert to TE, TM
    d = fdp.polarisation_mode_decomposition(ux, uy, Ex, Ey, Ez, 1)

    print(d["TE"][0, 1], d["TM"][0, 1])

    # check one value
    assert np.abs(0.9128709291752768 - 0.40824829046386296j - d["TE"][0, 1]) < 1e-5
    assert np.abs(-0.408248290463863 - 0.9128709291752767j - d["TM"][0, 1]) < 1e-5

    for iy in range(5):
        for ix in range(5):
            # check unit amplitude
            assert np.abs(d["TE"][iy, ix]) > 0.99999
            assert np.abs(d["TE"][iy, ix]) < 1.00001
            assert np.abs(d["TM"][iy, ix]) > 0.99999
            assert np.abs(d["TM"][iy, ix]) < 1.00001
            # and check 90 deg phase
            assert np.abs(d["TM"][iy, ix] / d["TE"][iy, ix] + 1j) < 1e-5


def test_arcoat():
    """Test for AR coating function."""

    # make grid
    s_ = -0.2 * np.arange(-2, 3)
    ux, uy = np.meshgrid(s_, s_)
    nl = 16
    ll = np.linspace(0.5, 2.0, nl)
    sgn = 1.0

    # simple case
    arcoat = fdp.FilterDetector([1.5, 2.0], [0.3, 0.6], sgn)
    for j in range(nl):
        Ex1, Ey1, Ez1 = arcoat.transmitted_E(ll[j], ux, uy, 2.0)
        assert np.abs(Ez1[2, 2, 0]) < 1.0e-2

    print(Ex1[0, 1, 0], Ey1[0, 1, 0], Ez1[0, 1, 0])

    # regression test
    assert np.abs(615039126.5069386 + 451425661.46545035j - Ex1[0, 1, 0]) < 100.0
    assert np.abs(-1801656306.183906 - 1331012972.7661047j - Ey1[0, 1, 0]) < 100.0
    assert np.abs(187252445.03809273 + 123562816.98367552j - Ez1[0, 1, 0]) < 100.0

    # check that if we split a layer we get the same answer
    arcoat2 = fdp.FilterDetector([1.5, 2.0, 2.0], [0.3, 0.3, 0.3], sgn)
    Ex2, Ey2, Ez2 = arcoat2.transmitted_E(ll[-1], ux, uy, 2.0)
    assert np.all(np.abs(Ex1 - Ex2) < 1.0)
    assert np.all(np.abs(Ey1 - Ey2) < 1.0)
    assert np.all(np.abs(Ez1 - Ez2) < 1.0)
