"""Test functions for the filter."""

import numpy as np
from psfsim import filter_detector_properties as fdp


def test_rotation():
    """Test of rotation matrices."""

    # Rotation of longitude between FPA (x,y,z) and plane of incidence coordinates.
    s_ = -0.2 * np.arange(-2, 3)
    ux, uy = np.meshgrid(s_, s_)
    RT = fdp.local_to_FPA_rotation(ux, uy, 1)
    assert np.shape(RT) == (5, 5, 3, 3)
    assert np.all(
        np.abs(
            RT[0, 1, :, :]
            - np.array([[0.89442719, 0.4472136, 0.0], [-0.4472136, 0.89442719, 0.0], [0.0, 0.0, 1.0]])
        )
        < 1e-5
    )
    RT = fdp.local_to_FPA_rotation(ux, uy, -1)
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
