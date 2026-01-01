"""Test functions for psfobject.py."""

import numpy as np
from psfsim.psfobject import PSFObject


def test_psfobject():
    """Test function for PSF object."""

    n = 8

    obj = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=n,
        npix_boundary=1,
        use_postage_stamp_size=False,
        add_focus=None,
    )

    assert np.abs(obj.dx - 10.0 / n) < 1.0e-3

    print(obj.ulen)
    # assert obj.npix_boundary == -1 # <-- used to force failure so we can look at the logs
