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
        xfpa, yfpa = wct._from_angle_to_fpa(row[2], row[3], wavelength=0.48)
        assert np.abs(xfpa - row[0]) < 2.0
        assert np.abs(yfpa - row[1]) < 2.0
        xfpa2, yfpa2 = wct.from_angle_to_fpa(row[2], row[3])
        assert np.hypot(xfpa2 - xfpa, yfpa2 - yfpa) < 0.2

        xan, yan = wct._from_fpa_to_angle((row[0], row[1]), wavelength=0.48)
        assert np.abs(xan - row[2]) < 0.006
        assert np.abs(yan - row[3]) < 0.006


def test_mapping_0055():
    """Test FPA->field against RST-SYS-0055E."""

    # points A-Z
    points = [
        [0.0, -103.5, -0.000000000000027, 0.301779571406688],
        [-44.1, -103.5, -0.132336451035948, 0.301144502989866],
        [-44.1, -94.5, -0.132591636036753, 0.275612629635079],
        [-89.5, -94.5, -0.268418757238188, 0.273697295943995],
        [-89.5, -72.5, -0.269612402897836, 0.210823665713674],
        [-135.0, -72.5, -0.404960581115742, 0.207863294373705],
        [-135.0, -28.5, -0.40809795322994, 0.080262535093205],
        [-133.9, 20.7, -0.407616388406991, -0.065506388863184],
        [-133.1, 64.7, -0.407039906847527, -0.198121477767259],
        [-89.3, 64.7, -0.274263185008575, -0.196864599042315],
        [-89.3, 43.4, -0.273716406807159, -0.13216009028812],
        [-45.0, 43.4, -0.138293208288634, -0.131216663324489],
        [-45.0, 34.7, -0.138164232496176, -0.10483920351578],
        [0.0, 34.7, -0.000000000000033, -0.104493140305456],
        [45.0, 34.7, 0.138164232496159, -0.104839203515821],
        [45.0, 43.4, 0.138293208288608, -0.131216663324511],
        [89.3, 43.4, 0.273716406807493, -0.132160090288021],
        [89.3, 64.7, 0.274263185009286, -0.196864599042073],
        [133.1, 64.7, 0.407039906852611, -0.198121477766049],
        [133.9, 20.7, 0.407616388416403, -0.065506388859912],
        [135.0, -28.5, 0.408097953268339, 0.080262535110754],
        [135.0, -72.5, 0.404960581236054, 0.207863294440494],
        [89.5, -72.5, 0.269612402920166, 0.210823665732642],
        [89.5, -94.5, 0.268418757282703, 0.273697295984906],
        [44.1, -94.5, 0.132591636042993, 0.275612629646861],
        [44.1, -103.5, 0.132336451044518, 0.301144503006441],
    ]

    for point in points:
        xan, yan = wct.from_fpa_to_angle((point[0], point[1]), ray_trace=True, use_filter="Y")
        err = np.hypot(xan - point[2], yan - point[3])
        print(point, xan, yan, err)
        assert err < 4.0e-5


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
