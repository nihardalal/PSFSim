"""Test for distortion maps."""

import numpy as np
from psfsim import wfi_coordinate_transformations as wct

grid = np.array(
    [[-22.14, 12.15, -0.071, -0.037], [-131.14, 62.64, -0.403, -0.191], [-133.08, -71.5, -0.405, 0.208]]
)


def test_mapping():
    """First-pass angle <-> FPA mapping test."""

    for r in range(np.shape(grid)[0]):
        row = grid[r]
        xfpa, yfpa = wct.from_angle_to_fpa(row[2], row[3], wavelength=0.48)
        assert np.abs(xfpa - row[0]) < 2.0
        assert np.abs(yfpa - row[1]) < 2.0

        xan, yan = wct.from_fpa_to_angle((row[0], row[1]), wavelength=0.48)
        assert np.abs(xan - row[2]) < 0.006
        assert np.abs(yan - row[3]) < 0.006


def test_sca():
    """Test for SCA <-> FPA and SCA <-> analysis mappings."""

    d = 20.44
    xfpa, yfpa = wct.from_sca_to_fpa(1, 0.0, 0.0)
    assert np.hypot(-22.14 - xfpa, 12.15 - yfpa) < 0.02
    xfpa, yfpa = wct.from_sca_to_fpa(1, d, -d)
    assert np.hypot(-42.58 - xfpa, 32.59 - yfpa) < 0.02
    xfpa, yfpa = wct.from_sca_to_fpa(1, d, d)
    assert np.hypot(-42.58 - xfpa, -8.29 - yfpa) < 0.02
    xfpa, yfpa = wct.from_sca_to_fpa(1, -d, d)
    assert np.hypot(-1.7 - xfpa, -8.29 - yfpa) < 0.02
    xfpa, yfpa = wct.from_sca_to_fpa(1, -d, -d)
    assert np.hypot(-1.7 - xfpa, 32.59 - yfpa) < 0.02

    x_ay, y_ay = wct.from_sca_to_analysis(4, 1.0, 0.0)
    assert np.hypot(-1000.0 - x_ay, 0.0 - y_ay) < 1.0
    x_ay, y_ay = wct.from_sca_to_analysis(4, 0.0, 1.0)
    assert np.hypot(0.0 - x_ay, -1000.0 - y_ay) < 1.0
