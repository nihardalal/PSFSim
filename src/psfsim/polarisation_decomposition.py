""" Functions to handle decomposition of (un)polarised E fields into TE and TM modes and rotation of E field components from local coordinates to focal plane coordinates. """


import time
import numpy as np
from numpy import newaxis as na

def local_to_fpa_rotation(ux, uy, sgn):
    """
    Local --> FPA rotation for electric field.

    This constructs an array of 3x3 rotation matrices from {incident ray in yz-plane}
    --> {focal plane coordinates}. The z-axis is perpendicular to the detector surface
    in both cases.

    By construction, this function has a discontinuity at u=0. It defaults to 0
    for unphysical rays (ux, uy outside the unit circule).

    Parameters
    ----------
    ux, uy : np.ndarray of float
         Orthographic projection of ray directions (each component). Should
         be the same shape.
    sgn : float
         Whether to flip z-direction for diagonal rays; should be +1 or -1.

    Returns
    -------
    RT : np.ndarray of float
         Shape is shape of ux + (3, 3). Each entry ends in a rotation matrix.

    """

    # print("Computing local to FPA rotation.......")
    # start_time = time.time()
    u = np.sqrt((ux**2) + (uy**2))
    mask = u <= 1
    # mask = np.abs(ux) + np.abs(uy) <= 1
    try:
        shape = ux.shape
    except AttributeError:
        shape = (1, 1)
    RT = np.zeros(shape + (3, 3), dtype=np.float64)

    # RT[mask & (u == 0)] = np.identity(3)
    # RT[mask & (u != 0), 0, 0] = uy[mask & (u != 0)] * sgn / u[mask & (u != 0)]
    # RT[mask & (u != 0), 0, 1] = ux[mask & (u != 0)] / u[mask & (u != 0)]
    # RT[mask & (u != 0), 1, 0] = -(ux[mask & (u != 0)] * sgn / u[mask & (u != 0)])
    # RT[mask & (u != 0), 1, 1] = uy[mask & (u != 0)] / u[mask & (u != 0)]
    # RT[mask & (u != 0), 2, 2] = sgn

    # alternate, somewhat streamlined version of code
    psi = np.arctan2(uy[mask], ux[mask])
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    RT[mask, 0, 0] = sgn * sinpsi
    RT[mask, 0, 1] = cospsi
    RT[mask, 1, 0] = -sgn * cospsi
    RT[mask, 1, 1] = sinpsi
    RT[mask, 2, 2] = sgn

    # end_time = time.time()
    # print(f"Finished computing local to FPA rotation in {end_time-start_time:.3f}")

    return RT


def polarisation_mode_decomposition(ux, uy, E, sgn):
    """
    Decomposes incident electric field (specified by components along FPA axes) into TE and TM modes.

    The convention is that the "TE" (S) direction is given by
    {propagation of ray} cross {original z axis} and the "TM" (P) direction is in the plane of incidence,
    pointed toward sgn * {original z axis}.

    Parameters
    ----------
    ux, uy : np.ndarray of float
        The orthographic projection of directions of propagation, each is an array.
    E : np.ndarray of complex
        The incident electric field. Same shape as `ux.shape + (3,)`.
    sgn : float
        Whether the z-direction of the initial coordinate system should be the same (+1) or
        opposite (-1) the AR coating coordinate system.

    Returns
    -------
    dict of np.ndarray of complex
        The keys are "TE" and "TM", and each have the same shape as `ux` and `uy`.

    See Also
    --------
    local_to_fpa_rotation
        This function is used for the convention when (ux, uy) == (0, 0).

    """
    Ex = E[:, :, 0]
    Ey = E[:, :, 1]
    Ez = E[:, :, 2]

    u = np.sqrt((ux**2) + (uy**2))
    try:
        shape = ux.shape
    except AttributeError:
        shape = (1, 1)
    mask = u <= 1
    # mask = np.abs(ux) + np.abs(uy) <= 1

    # get rotation matrix ()
    RT = local_to_fpa_rotation(ux, uy, sgn)
    wmask = np.sqrt(1 - u[mask] ** 2)

    A_TE = np.zeros(shape, dtype=np.complex128)
    A_TM = np.zeros(shape, dtype=np.complex128)

    # A_TE[mask & (u == 0)] = Ex[mask & (u == 0)]
    # A_TM[mask & (u == 0)] = -Ey[mask & (u == 0)]

    ee1 = np.zeros_like(ux, dtype=np.complex128)
    ee2 = np.zeros_like(ux, dtype=np.complex128)

    ek1 = np.zeros_like(ux, dtype=np.complex128)
    ek2 = np.zeros_like(ux, dtype=np.complex128)
    ek3 = np.zeros_like(ux, dtype=np.complex128)

    # ek1[mask & (u != 0)] = -(ux[mask & (u != 0)] / u[mask & (u != 0)]) * np.sqrt(
    #     1 - (u[mask & (u != 0)] ** 2)
    # )
    # ek2[mask & (u != 0)] = -(uy[mask & (u != 0)] / u[mask & (u != 0)]) * np.sqrt(
    #     1 - (u[mask & (u != 0)] ** 2)
    # )
    # ek3[mask & (u != 0)] = u[mask & (u != 0)] * sgn

    ee1[mask] = RT[mask, 1, 1]
    ee2[mask] = -RT[mask, 0, 1]

    ek1[mask] = -RT[mask, 0, 1] * wmask
    ek2[mask] = -RT[mask, 1, 1] * wmask
    ek3[mask] = u[mask] * sgn

    # A_TE[mask & (u != 0)] = (Ex[mask & (u != 0)] * (uy[mask & (u != 0)] / u[mask & (u != 0)])) - (
    #     Ey[mask & (u != 0)] * (ux[mask & (u != 0)] / u[mask & (u != 0)])
    # )
    # A_TM[mask & (u != 0)] = (
    #     (ek1[mask & (u != 0)] * Ex[mask & (u != 0)])
    #     + (ek2[mask & (u != 0)] * Ey[mask & (u != 0)])
    #     + (ek3[mask & (u != 0)] * Ez[mask & (u != 0)])
    # )

    A_TE[:, :] = Ex * ee1 + Ey * ee2
    A_TM[:, :] = Ex * ek1 + Ey * ek2 + Ez * ek3

    return {"TE": A_TE, "TM": A_TM}


def unpolarised_mode_decomposition(ux, uy, E0=1.0e10):
    """
    Decomposition for unpolarized light.

    Parameters
    ----------
    ux, uy : float
        Orthographic coordinates.
    E0 : float, optional
        Amplitude.

    Returns
    -------
    dict of np.ndarray of complex
        The keys are "TE" and "TM", and each have the same shape as `ux` and `uy`.

    Notes
    -----
    **Warning** : This generates equal amplitudes but if you just use the field
    values there will be unphysical interference. Need to figure out what to do about this.

    """

    print("Computing polarisation mode decomposition for unpolarised incident E field.....")
    start_time = time.time()

    # Function to obtain TE and TM mode amplitudes for unpolarised incident wave with magnitude of
    # electric field E0.

    u = np.sqrt((ux**2) + (uy**2))
    mask = u <= 1
    # mask = np.abs(ux) + np.abs(uy) <= 1.0
    try:
        shape = ux.shape
    except AttributeError:
        shape = (1, 1)
    # shape = ux.shape
    A_TE = np.zeros(shape, dtype=np.complex128)
    A_TM = np.zeros(shape, dtype=np.complex128)

    A_TE[mask] = (1.0 / np.sqrt(2)) * E0
    A_TM[mask] = -(1.0 / np.sqrt(2)) * E0

    end_time = time.time()
    print(f"Finished computing polarisation mode decomposition in {end_time-start_time:.3f}")
    return {"TE": A_TE, "TM": A_TM}

