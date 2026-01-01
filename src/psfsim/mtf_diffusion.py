"""Detector MTF functions."""

import numpy as np
from numba import njit, prange
from numpy import newaxis as na
from scipy.signal import fftconvolve
from scipy.special import erf

# Pixel properties
from .wfi_data import cdpar, nside, pix


def diffusion_green(xd, yd, x2=0.0, y2=0.0):
    """
    Charge diffusion Green's function as a function of Analysis coordinates in mm.

    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA,
    where the parameters are derived from the charge diffusion model
    described in Emily Macbeth's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632.

    Parameters
    ----------
    xd, yd : float or np.ndarray of float
        The coordinates where the Green's function is to be computed.
        Must be the same shape, or broadcastable to the same shape.
    x2, y2 : float, optional
        The location where the charges are generated.

    Returns
    -------
    float or np.ndarray of float
        The Green's function at the indicated positions, same shape as `xd`.

    """

    sigma_s = cdpar["sigma_s"]

    sq = (xd - x2) ** 2 + (yd - y2) ** 2

    term = 0.0
    for j in range(len(cdpar["w"])):
        sigmaj = cdpar["c"][j] * sigma_s
        term += cdpar["w"][j] * np.exp(-(sq / (2.0 * sigmaj**2))) * (1.0 / (2.0 * np.pi * sigmaj**2))

    return term


def diffusion_prob(xd, yd, width=10.0, x2=0.0, y2=0.0):
    """
    Charge diffusion probability as a function of Analysis coordinates in mm.

    This is the integral of the Green's function over a square of width `width` centered on (0, 0).

    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA,
    where the parameters are derived from the charge diffusion model
    described in Emily Macbeth's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632.

    Parameters
    ----------
    xd, yd : float or np.ndarray of float
        The coordinates where the Green's function is to be computed.
        Must be the same shape, or broadcastable to the same shape.
    width : float, int
        The width of the integration zone in microns (on each axis).
    x2, y2 : float, optional
        The location where the charges are generated.

    Returns
    -------
    float or np.ndarray of float
        The integrated probability centered at the indicated positions, same shape as `xd`.

    See Also
    --------
    diffusion_green
        The probability density function.

    """

    sigma_s = cdpar["sigma_s"]

    cut_xm = xd - x2 - width / 2.0
    cut_xp = xd - x2 + width / 2.0
    cut_ym = yd - y2 - width / 2.0
    cut_yp = yd - y2 + width / 2.0

    term = 0.0
    for j in range(len(cdpar["w"])):
        sigmaj2 = cdpar["c"][j] * sigma_s * np.sqrt(2.0)
        term += (
            0.25
            * cdpar["w"][j]
            * (erf(cut_xp / sigmaj2) - erf(cut_xm / sigmaj2))
            * (erf(cut_yp / sigmaj2) - erf(cut_ym / sigmaj2))
        )

    return term


def intensity_to_image(intensity, x_in, y_in, x_out, y_out, n_out, dx, reflect=True, tophat=True):
    """
    Function to convolve an intensity image with the Green's function.

    Parameters
    ----------
    intensity : np.ndarray of float
        The intensity image.
    x_in, y_in : float
        The center of the intensity input image in microns.
    x_out, y_out: float
        The center of the desired output image in microns.
    n_out : int
        The desired output postage stamp size.
    dx : float
        The grid spacing in microns.
    reflect : bool, optional
        Reflect at the detector active region edge.
    tophat : bool, optional
        Integrate over the pixel tophat. Note that if this option is chosen, the unit of
        the output is the unit of the intensity times micron^2.

    Returns
    -------
    np.ndarray of float
        The output image.

    """

    (ny_in, nx_in) = np.shape(intensity)

    # List of positions to reflect.
    pos = [(0, 0)]
    if reflect:
        x_sgn = 1 if x_in > 0 else -1
        y_sgn = 1 if y_in > 0 else -1
        pos = [(0, 0), (x_sgn, 0), (0, y_sgn), (x_sgn, y_sgn)]
    hds = pix * nside / 2.0  # coordinate of side

    # Sum over reflected images
    im_out = np.zeros((n_out, n_out))
    for p in pos:
        ii = np.copy(intensity)
        if p[0] % 2 != 0:
            ii = np.fliplr(ii)
        if p[1] % 2 != 0:
            ii = np.flipud(ii)
        x_refl = p[0] * hds + (-1) ** p[0] * (x_in - p[0] * hds)
        y_refl = p[1] * hds + (-1) ** p[1] * (y_in - p[1] * hds)

        # Now do the convolution
        delta_x_ctr = x_out - x_refl
        delta_y_ctr = y_out - y_refl
        nyvals = ny_in + n_out - 1
        nxvals = nx_in + n_out - 1
        dyvals = np.linspace(
            delta_y_ctr - dx * (nyvals - 1.0) / 2.0, delta_y_ctr + dx * (nyvals - 1.0) / 2.0, nyvals
        )
        dxvals = np.linspace(
            delta_x_ctr - dx * (nxvals - 1.0) / 2.0, delta_x_ctr + dx * (nxvals - 1.0) / 2.0, nxvals
        )
        if tophat:
            g_offset = diffusion_prob(dxvals[None, :], dyvals[:, None], width=pix)
        else:
            g_offset = diffusion_green(dxvals[None, :], dyvals[:, None])
        im_out[:, :] += fftconvolve(g_offset, ii, mode="valid")

    return im_out


### not sure if we need things after here

MTF = diffusion_green  # alias, will delete if not needed later


# def MTF_array(pixelsampling=1.0, ps=6):
#
#     Function to compute the profile of the modulation tranfer function in analysis coordinates at a
#     spacing of PSFObject.dsX microns over a grid of size ps*pix x ps*pix microns, where pix is the
#     native pixel size of 10 microns.
#
#     pix = 10  # pixel size in microns
#     sigma_s = 0.3279 * pix  # sigma of the charge diffusion in pixel units
#     sigma_s = sigma_s
#     w1 = 0.17519
#     w2 = 0.53146
#     w3 = 0.29335
#     c1 = 0.4522
#     c2 = 0.8050
#     c3 = 1.4329
#
#     sigma1 = c1 * sigma_s
#     sigma2 = c2 * sigma_s
#     sigma3 = c3 * sigma_s

#     xd = np.arange(-ps * pix / 2, (ps * pix / 2) + pixelsampling, pixelsampling)
#     yd = np.arange(-ps * pix / 2, (ps * pix / 2) + pixelsampling, pixelsampling)
#     Xd, Yd = np.meshgrid(xd, yd, indexing="ij")
#     MTF1 = w1 * np.exp(-((Xd**2 + Yd**2) / (2 * sigma1**2))) * (1 / (2 * np.pi * sigma1**2))
#     MTF2 = w2 * np.exp(-((Xd**2 + Yd**2) / (2 * sigma2**2))) * (1 / (2 * np.pi * sigma2**2))
#     MTF3 = w3 * np.exp(-((Xd**2 + Yd**2) / (2 * sigma3**2))) * (1 / (2 * np.pi * sigma3**2))
#     MTF_total = MTF1 + MTF2 + MTF3
#     return (Xd, Yd, MTF_total)


def diffusion_green_image(xd, yd, sX, sY, intensity, npix_boundary=1):
    """
    Function to compute the detector response at the detector on (SCAx, SCAy) from an image with
    analysis coordinates sX, sY (meshgrid) and the intensity profile given by Intensity.

    sX, sY : ulen x ulen meshgrid arrays of the image coordinates (in analysis coordinates, in mm)
    xd, yd : x and y coordinates (in the Analysis coordinate system) of the postage stamp point
    intensity : ulen x ulen array of intensity values integrated over the depth of the detector.

    """
    pix = 10  # pixel size in microns
    nside = 4088  # number of active pixels per side in the SCA (Should this be 4088 or 4096?)
    side_length = nside * pix  # Length of the SCA in mm
    # xp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # x coordinates of pixel centers in mm
    # yp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # y coordinates of pixel centers in mm
    # Xp, Yp = np.meshgrid(xp_array, yp_array, indexing='ij')
    # Create a meshgrid of pixel centers
    # MTF_SCA_array = np.zeros(x.shape+(nside, nside))
    dsX = sX[1, 0] - sX[0, 0]
    dsY = sY[0, 1] - sY[0, 0]

    MTF_array = np.zeros(sX.shape, dtype=np.float64)

    SCA_mask = np.maximum(np.abs(sX), np.abs(sY)) <= side_length / 2
    left_mask = SCA_mask & (sX < -side_length / 2 + npix_boundary * pix)
    right_mask = SCA_mask & (sX > side_length / 2 - npix_boundary * pix)
    bottom_mask = SCA_mask & (sY < -side_length / 2 + npix_boundary * pix)
    top_mask = SCA_mask & (sY > side_length / 2 - npix_boundary * pix)

    MTF_array[SCA_mask] += MTF(sX[SCA_mask], sY[SCA_mask], xd, yd) * intensity[SCA_mask]
    MTF_array[left_mask] += (
        MTF(sX[left_mask] - 2 * (sX[left_mask] + side_length / 2), sY[left_mask], xd, yd)
        * intensity[left_mask]
    )
    MTF_array[right_mask] += (
        MTF(sX[right_mask] + 2 * (side_length / 2 - sX[right_mask]), sY[right_mask], xd, yd)
        * intensity[right_mask]
    )
    MTF_array[top_mask] += (
        MTF(sX[top_mask], sY[top_mask] + 2 * (side_length / 2 - sY[top_mask]), xd, yd) * intensity[top_mask]
    )
    MTF_array[bottom_mask] += (
        MTF(sX[bottom_mask], sY[bottom_mask] - 2 * (sY[bottom_mask] + side_length / 2), xd, yd)
        * intensity[bottom_mask]
    )

    return np.sum(MTF_array * dsX * dsY)


MTF_image = diffusion_green_image  # alias, will delete if not needed later


def MTF_image_vec(psX, psY, sX, sY, intensity, npix_boundary=1):
    """
    Function to compute the detector response at the detector on (SCAx, SCAy) from an image with
    analysis coordinates sX, sY (meshgrid) and the intensity profile given by Intensity.

    sX, sY : ulen x ulen meshgrid arrays of the image coordinates (in analysis coordinates, in mm)
    psX, psY : postage_stamp_size x postage_stamp_size meshgrid of the coordinates (in the Analysis
    coordinate system) of the postage stamp points
    intensity : ulen x ulen array of intensity values integrated over the depth of the detector.

    """
    pix = 10  # pixel size in microns
    nside = 4088  # number of active pixels per side in the SCA
    side_length = nside * pix  # Length of the SCA in mm
    # xp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # x coordinates of pixel centers in mm
    # yp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # y coordinates of pixel centers in mm
    # Xp, Yp = np.meshgrid(xp_array, yp_array, indexing='ij')  # Create a meshgrid of pixel centers
    # MTF_SCA_array = np.zeros(x.shape+(nside, nside))
    dsX = sX[1, 0] - sX[0, 0]
    dsY = sY[0, 1] - sY[0, 0]

    MTF_array = np.zeros(psX.shape + sX.shape, dtype=np.float64)

    mask_PS = np.maximum(np.abs(psX), np.abs(psY)) <= side_length / 2
    SCA_mask = np.maximum(np.abs(sX), np.abs(sY)) <= side_length / 2
    left_mask = SCA_mask & (sX < -side_length / 2 + npix_boundary * pix)
    right_mask = SCA_mask & (sX > side_length / 2 - npix_boundary * pix)
    bottom_mask = SCA_mask & (sY < -side_length / 2 + npix_boundary * pix)
    top_mask = SCA_mask & (sY > side_length / 2 - npix_boundary * pix)

    MTF_array[mask_PS, SCA_mask] += (
        MTF(sX[SCA_mask][na, :], sY[SCA_mask][na, :], psX[mask_PS][:, na], psY[mask_PS][:, na])
        * intensity[SCA_mask][na, :]
    )
    MTF_array[mask_PS, left_mask] += (
        MTF(
            sX[na, left_mask] - 2 * (sX[na, left_mask] + side_length / 2),
            sY[na, left_mask],
            psX[mask_PS, na],
            psY[mask_PS, na],
        )
        * intensity[na, left_mask]
    )
    MTF_array[mask_PS, right_mask] += (
        MTF(
            sX[na, right_mask] + 2 * (side_length / 2 - sX[na, right_mask]),
            sY[na, right_mask],
            psX[mask_PS, na],
            psY[mask_PS, na],
        )
        * intensity[na, right_mask]
    )
    MTF_array[mask_PS, top_mask] += (
        MTF(
            sX[na, top_mask],
            sY[na, top_mask] + 2 * (side_length / 2 - sY[na, top_mask]),
            psX[mask_PS, na],
            psY[mask_PS, na],
        )
        * intensity[na, top_mask]
    )
    MTF_array[mask_PS, bottom_mask] += (
        MTF(
            sX[na, bottom_mask],
            sY[na, bottom_mask] - 2 * (sY[na, bottom_mask] + side_length / 2),
            psX[mask_PS, na],
            psY[mask_PS, na],
        )
        * intensity[na, bottom_mask]
    )

    return np.sum(MTF_array * dsX * dsY, axis=(-2, -1))


@njit
def MTF_SCA(psX, psY, x, y, npix_boundary=1):
    """
    psX, psY : postage_stamp_size x postage_stamp_size meshgrid of the coordinates (in the Analysis
    coordinate system) of the postage stamp points

    x, y : Analysis coordinates of the point in the SCA from which the diffusion is computed (in microns)

    npix_boundary : number of pixels in the boundary layer where reflection boundary conditions are applied

    """
    pix = 10  # pixel size in microns
    nside = 4088  # number of active pixels per side in the SCA (Should this be 4088 or 4096?)
    side_length = nside * pix  # Length of the SCA in micron
    # xp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # x coordinates of pixel centers in mm
    # yp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)
    # y coordinates of pixel centers in mm
    # Xp, Yp = np.meshgrid(xp_array, yp_array, indexing='ij')  # Create a meshgrid of pixel centers
    MTF_SCA_array = np.zeros(psX.shape, dtype=np.float64)

    if not max(np.abs(x), np.abs(y)) < side_length / 2:
        # print('point is outside the SCA')
        return MTF_SCA_array

    else:
        MTF_SCA_array = MTF(psX, psY, x, y)  ## Baseline MTF without any boundary layer corrections

        # Check if the point is in the left/right boundary layer and apply reflection boundary conditions
        # if necessary
        if x < -side_length / 2 + npix_boundary * pix:
            # point is in the left boundary layer
            x_reflected = x - 2 * (x + side_length / 2)
            MTF_SCA_array += MTF(psX, psY, x_reflected, y)

        elif x > side_length / 2 - npix_boundary * pix:
            # point is in the right boundary layer
            x_reflected = x + 2 * (side_length / 2 - x)
            MTF_SCA_array += MTF(psX, psY, x_reflected, y)

        # Check if the point is in the top/bottom boundary layer and apply reflection boundary conditions
        # if necessary
        if y < -side_length / 2 + npix_boundary * pix:
            # point is in the bottom boundary layer
            y_reflected = y - 2 * (y + side_length / 2)
            MTF_SCA_array += MTF(psX, psY, x, y_reflected)

        elif y > side_length / 2 - npix_boundary * pix:
            # point is in the top boundary layer
            y_reflected = y + 2 * (side_length / 2 - y)
            MTF_SCA_array += MTF(psX, psY, x, y_reflected)

        return MTF_SCA_array


@njit(parallel=True)
def MTF_SCA_postage_stamp(x, y, psX, psY, intensity, npix_boundary=1):
    """
    Function to compute the detector response at the detector on (SCAx, SCAy) from an image with
    analysis coordinates sX, sY (meshgrid) and the intensity profile given by Intensity.

    x, y : arrays of the analysis coordinates (in mm) over which the intensity is defined.
    Each is a 1D array with len=ulen

    psX, psY : postage_stamp_size x postage_stamp_size meshgrid of the coordinates (in the Analysis
       coordinate system) of the postage stamp points

    intensity : ulen x ulen array of intensity values integrated over the depth of the detector.

    """
    nx = x.shape[0]
    ny = y.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    out_shape = psX.shape
    # Create an accumulator for each iteration
    # temp = np.zeros((nx * ny, out_shape[0], out_shape[1]), dtype=np.float64)
    result = np.zeros(psX.shape, dtype=np.float64)
    for i in prange(nx * ny):
        ix = i // ny
        iy = i % ny
        # temp[i, :, :] = intensity[ix, iy] * MTF_SCA(x[ix], y[iy], psX, psY, npix_boundary) * dx * dy
        temp = intensity[ix, iy] * MTF_SCA(psX, psY, x[ix], y[iy], npix_boundary) * dx * dy
        for index_x in range(out_shape[0]):
            for index_y in range(out_shape[1]):
                result[index_x, index_y] += temp[index_x, index_y]
    # Sum over all iterations to get the final result
    # result = np.sum(temp, axis=0)

    return result
