"""Utilities for WFI coordinate systems."""

import numpy as np

from . import wfi_data


def from_angle_to_fpa(xan, yan, wavelength=0.48):
    """
    Coarse transformation from field angle to FPA position.

    Parameters
    ----------
    xan, yan : float
        Field positions in degrees.
    wavelength : float, optional
        Vacuum wavelength in microns.

    Returns
    -------
    (float, float)
         Focal plane position in mm.

    """

    # xan, yan in degrees, wavelength in micrometers
    wavindex = np.argmin(wfi_data.wavelength - wavelength)
    coeff = wfi_data.angle_to_fpa_poly_coefficients[wavindex]
    powers = xan ** wfi_data.exponents[:, 0] * yan ** wfi_data.exponents[:, 1]
    xterms = coeff[:, 0] * powers
    yterms = coeff[:, 1] * powers
    return (np.sum(xterms), np.sum(yterms))


def from_fpa_to_angle(fpapos, wavelength=0.48):
    """
    Coarse transformation from FPA position to field angle.

    Parameters
    ----------
    fpapos : (float, float)
         Focal plane position in mm.
    wavelength : float, optional
        Vacuum wavelength in microns.

    Returns
    -------
    (float, float)
        Field positions in degrees.

    """

    # FPAx, FPAy in mm, wavelength in micrometers
    FPAx = fpapos[0]
    FPAy = fpapos[1]
    wavindex = np.argmin(wfi_data.wavelength - wavelength)
    coeff = wfi_data.fpa_to_angle_poly_coefficients[wavindex]
    powers = FPAx ** wfi_data.exponents[:, 0] * FPAy ** wfi_data.exponents[:, 1]
    xterms = coeff[:, 0] * powers
    yterms = coeff[:, 1] * powers
    return (np.sum(xterms), np.sum(yterms))


def from_sca_to_fpa(scanum, scax, scay):
    """
    Coordinate transformation converting SCA to FPA coordinates.

    The "SCA" coordinates are aligned with the SOC "Science" frame (i.e., 180 degrees
    rotated relative to the FPA frame).

    Parameters
    ----------
    scanum : int
        The SCA number (1 through 18).
    scax, scay : float
        SCA coordinates in mm: (0, 0) is the center.

    Returns
    -------
    (float, float)
        The FPA coordinates in mm.

    """

    xfpa = np.array(
        [
            -22.14,
            -22.29,
            -22.44,
            -66.42,
            -66.92,
            -67.42,
            -110.70,
            -111.48,
            -112.64,
            22.14,
            22.29,
            22.44,
            66.42,
            66.92,
            67.42,
            110.70,
            111.48,
            112.64,
        ]
    )
    yfpa = np.array(
        [
            12.15,
            -37.03,
            -82.06,
            20.90,
            -28.28,
            -73.06,
            42.20,
            -6.98,
            -51.06,
            12.15,
            -37.03,
            -82.06,
            20.90,
            -28.28,
            -73.06,
            42.20,
            -6.98,
            -51.06,
        ]
    )
    sc_index = scanum - 1
    # pixsize = 0.01
    # nside = 4088
    if np.amin(scanum) < 1 or np.amax(scanum) > 18:
        raise ValueError("Invalid SCA Number")
    return (xfpa[sc_index] - scax, yfpa[sc_index] - scay)


def from_sca_to_analysis(scanum, scax, scay):
    """
    Coordinate transformation converting SCA coordinates (in mm) to Analysis coordinates (in microns).

    The "SCA" coordinates are aligned with the SOC "Science" frame (i.e., 180 degrees
    rotated relative to the FPA frame).

    The Analysis coordinates system is defined to be the FPA coordinate system with origin shifted to
    the center of the SCA.

    Parameters
    ----------
    scanum : int
        The SCA number (1 through 18).
    scax, scay : float
        SCA coordinates in mm: (0, 0) is the center.

    Returns
    -------
    (float, float)
        The analysis coordinates in microns.

    """

    # sc_index = scanum - 1
    # pixsize = 10  # microns
    # nside = 4088
    if np.amin(scanum) < 1 or np.amax(scanum) > 18:
        raise ValueError("Invalid SCA Number")
    return (-1000.0 * scax, -1000.0 * scay)
