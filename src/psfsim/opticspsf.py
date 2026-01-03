"""Optics objects."""

from importlib.resources import files

import numpy as np
import pandas as pd
from astropy.io import fits

from . import wfi_data, zernike
from .romantrace import RomanRayBundle
from .wfi_coordinate_transformations import from_fpa_to_angle, from_sca_to_fpa


def altgriddata(points, values, xi):
    """
    Quadratic fitting function.

    This is a substitute for ``scipy.interpolate.griddata`` when we have 5 points
    and want to fit a quadratic, minus the x^2-y^2 term.

    Paramters
    ---------
    points : np.ndarray of float
        The coordinates of the 5 points where we have values; shape (5, 2).
    values : np.ndarray of flat
        The values to interpolate; shape (5,).
    xi : (float, float)
        The location to interpolate to.

    Returns
    -------
    float
        The interpolated value.

    """

    # Set up the linear system.
    # Order of coefficients is 1, x, y, x*y, x^2+y^2.
    bval = np.zeros((5, 5))  # bval[i,j] is the ith basis function at jth point

    bval[0, :] = 1.0
    bval[1, :] = points[:, 0]
    bval[2, :] = points[:, 1]
    bval[3, :] = points[:, 0] * points[:, 1]
    bval[4, :] = points[:, 0] ** 2 + points[:, 1] ** 2

    mat = bval @ bval.T
    vec = bval @ values

    coefs = np.linalg.solve(mat, vec)

    return (
        coefs[0]
        + coefs[1] * xi[0]
        + coefs[2] * xi[1]
        + coefs[3] * xi[0] * xi[1]
        + coefs[4] * (xi[0] ** 2 + xi[1] ** 2)
    )


def fit_linfunc(f):
    """
    Helper function to do a linear fit: f = a + b*x + c*y.

    Excludes nans. Here x and y start at 0.

    Parameters
    ----------
    f : np.ndarray of float
        The function to fit. Should be a 2D array, ``f[y, x]``.

    Returns
    -------
    coefs : np.ndarray of float
        The coefficients of the fit. Length 4: a, b, c, rmserr.

    """

    # get the x and y coordinates.
    x, y = np.meshgrid(
        np.arange(np.shape(f)[1]).astype(np.float64), np.arange(np.shape(f)[0]).astype(np.float64)
    )
    x = np.where(np.isfinite(f), x, np.nan)
    y = np.where(np.isfinite(f), y, np.nan)
    f = np.where(np.isfinite(f), f, np.nan)

    # set up linear system
    mat = np.zeros((3, 3))
    vec = np.zeros(3)
    mat[0, 0] = np.count_nonzero(np.isfinite(f))
    mat[0, 1] = mat[1, 0] = np.nansum(x)
    mat[0, 2] = mat[2, 0] = np.nansum(y)
    mat[1, 1] = np.nansum(x**2)
    mat[2, 2] = np.nansum(y**2)
    mat[1, 2] = mat[2, 1] = np.nansum(x * y)
    vec[0] = np.nansum(f)
    vec[1] = np.nansum(x * f)
    vec[2] = np.nansum(y * f)

    # fill solution
    coefs = np.zeros(4)
    coefs[:3] = np.linalg.solve(mat, vec)
    err = f - coefs[0] - coefs[1] * x - coefs[2] * y
    coefs[-1] = np.sqrt(np.nansum(err**2) / mat[0, 0])

    return coefs


def compute_jacobian(u, dx=1.0, dy=1.0):
    """
    Computes the Jacobian for entrance --> exit pupil.

    Parameters
    ----------
    u : np.ndarray of float
        3D array of exit pupil positions; ``u[iy, ix, ic]`` is the orthographic
        direction of the outgoing ray in entrance pixel (ix, iy). The components are
        ic == 0 for the x-component of `u` and ic == 1 for the y-component of `u`.
    dx, dy : float, optional
        The entrance pupil grid spacings.

    Returns
    -------
    np.ndarray of float
        The Jacobian, d(u_x,u_y)_out / d(x,y)_in. Shape is (`N`, `N`, 2, 2), where the
        first 2 axes refer to the shape of `u`, and the second 2 axes are matrix axes.

    """

    # Compute gradients for each component
    dux_dx, dux_dy = np.gradient(u[..., 0], dx, dy, axis=(1, 0))
    duy_dx, duy_dy = np.gradient(u[..., 1], dx, dy, axis=(1, 0))

    # Construct the Jacobian
    jacobian = np.empty((u.shape[0], u.shape[1], 2, 2))
    jacobian[..., 0, 0] = dux_dx  # d(u_x)/dx
    jacobian[..., 0, 1] = dux_dy  # d(u_x)/dy
    jacobian[..., 1, 0] = duy_dx  # d(u_y)/dx
    jacobian[..., 1, 1] = duy_dy  # d(u_y)/dy

    return jacobian


class GeometricOptics:
    """
    Geometric optics object.

    Parameters
    ----------
    scanum : int
        The SCA number.
    scax, scay : float
        The pixel positions on the SCA (in mm, FPA coordinates relative to the SCA center).
    wavelength : float, optional
        The vacuum wavelength in microns.
    use_filter : str
        The filter as a 1-character string.
    ulen : int, optional
        The size of array for pupil sampling.
    ray_trace : bool, optional
        Whether to use ray tracing.
    pixelsampling : float, optional
        Desired FFT-based output pixel sampling in microns.

    Attributes
    ----------
    wavelength : float
        The vacuum wavelength in microns.
    scanum : int
        The SCA number.
    scax, scay : float
        The pixel positions on the SCA (in mm, FPA coordinates relative to the SCA center).
    xan, yan : float
        The field angles in degrees.
    use_filter : str
        The filter as a 1-character string.
    ulen : int
        The FFT length (and pupil grid size).
    ucen, vcen : float
        The orthographic coordinates at the exit pupil of the center of the pupil image.
    du : float
        The grid spacing in orthographic coordinates at the exit pupil.
    path_difference : np.ndarray of float
        The wavefront map in microns. Shape (`ulen`, `ulen`).
    rb : psfsim.romantrace.RayBundle
        The ray trace object.

    Methods
    -------
    __init__
        Constructor.
    u_array
        Gets the 2D array of u.
    v_array
        Gets the 2D array of v.
    compute_distortion_matrix
        Computes the distortion matrix.
    compute_determinant
        Determinant of distortion matrix.
    load_pupil_mask
        Loads the pupil mask.
    path_diff
        Path difference map.

    """

    def __init__(
        self,
        scanum,
        scax,
        scay,
        wavelength=0.48,
        use_filter="H",
        ulen=2048,
        ray_trace=True,
        pixelsampling=1.00,
    ):
        # sca position in mm
        # wavelength in micrometers

        self.wavelength = wavelength
        self.dsX = pixelsampling  # pixel spacing in microns
        self.pupilLength = 2400 * 8  # in mm
        self.focalLength = 8  # m
        self.samplingwidth = (self.wavelength / self.dsX) * self.pupilLength  # in mm for raytrace

        self.scanum = scanum
        self.scax = scax
        self.scay = scay
        # print(self.scaX, self.scaY)

        self.xout, self.yout = from_sca_to_fpa(self.scanum, self.scax, self.scay)
        self.posOut = np.array([self.xout, self.yout])

        # Set up u,v array for computations of Zernicke Polynomials
        self.ulen = ulen

        # Go with some version of centered sampling if not using ray trace
        if not ray_trace:
            self.umin = (-0.5) * wavelength / self.dsX
            self.umax = (0.5) * wavelength / self.dsX
            self.uX = np.linspace(self.umin, self.umax, self.ulen)
            self.uY = np.linspace(self.umin, self.umax, self.ulen)
            self.uArray, self.vArray = np.meshgrid(self.uX, self.uY)

        self.pupilSampling = self.ulen

        # Get angular coordinates
        self.xan, self.yan = from_fpa_to_angle(
            self.posOut, wavelength=self.wavelength, ray_trace=ray_trace, use_filter=use_filter
        )

        self.use_filter = use_filter

        # Compute Distortion Matrix and dterminant
        self.distortionMatrix = self.compute_distortion_matrix(method="raytrace")
        self.determinant = self.compute_determinant()

        # Obtain pupil Mask
        self.pupil_mask = self.load_pupil_mask(use_ray_trace=ray_trace)
        u = np.where(self.pupil_mask > 0, self.pupil_mask_u[:, :, 0], np.nan)
        v = np.where(self.pupil_mask > 0, self.pupil_mask_u[:, :, 1], np.nan)
        self.uvcoefs = [fit_linfunc(u), fit_linfunc(v)]
        self.ucen = self.uvcoefs[0][0] + (self.uvcoefs[0][1] + self.uvcoefs[0][2]) * (self.ulen - 1.0) / 2.0
        self.vcen = self.uvcoefs[1][0] + (self.uvcoefs[1][1] + self.uvcoefs[1][2]) * (self.ulen - 1.0) / 2.0
        self.du = (self.uvcoefs[0][1] + self.uvcoefs[1][2]) / 2.0

        # Load pupil mask from raytrace - more accurate
        # self.uArray = self.pupilMaskU[:, :, 0]
        # self.vArray = self.pupilMaskU[:, :, 1]
        # self.umin = np.min(self.uArray)
        # self.umax = np.max(self.uArray)
        # self.vmin = np.min(self.vArray)
        # self.vmax = np.max(self.vArray)
        # self.ucen = 0.5 * (self.umin + self.umax)
        # self.vcen = 0.5 * (self.vmin + self.vmax)
        # Get path difference map
        self.path_difference = self.path_diff()

        # self.integrand = self.pupilMask*self.determinant\
        # *expm(2*np.pi/self.wavelength*1j*self.pathDifference)
        # self.x_minus = (-1)**np.array(range(ulen))#used to translate ftt to image center
        # self.ph = np.outer(self.x_minus, self.x_minus) #phase required to translate fft to center
        # self.eArray = np.fft.fft2(self.integrand*self.ph)
        # self.magEArray = abs(self.eArray)

    def u_array(self):
        """
        Gets the 2D array of u.
        """

        x, y = np.meshgrid(np.linspace(0, self.ulen - 1, self.ulen), np.linspace(0, self.ulen - 1, self.ulen))
        coefs = self.uvcoefs[0]
        return coefs[0] + coefs[1] * x + coefs[2] * y

    def v_array(self):
        """
        Gets the 2D array of u.
        """

        x, y = np.meshgrid(np.linspace(0, self.ulen - 1, self.ulen), np.linspace(0, self.ulen - 1, self.ulen))
        coefs = self.uvcoefs[1]
        return coefs[0] + coefs[1] * x + coefs[2] * y

    def compute_distortion_matrix(self, method="raytrace"):
        """
        Computes the distortion matrix.

        Parameters
        ----------
        method : str, optional
            If "poly", uses the pre-computed polynomial fit.
            The default (recommended) is "raytrace".

        Returns
        -------
        np.ndarray of float
            The 2x2 Jacobian matrix, d(xan,yan)/d(fpax,fpay); units of mm^-1.

        """

        if method == "poly":
            # Load in polynomial fits to Jacobian
            wavindex = np.argmin(wfi_data.wavelength - self.wavelength)
            coeff = wfi_data.angle_to_fpa_poly_coefficients[wavindex]
            xpowers = wfi_data.exponents[:, 0]
            ypowers = wfi_data.exponents[:, 1]
            newpowersx = np.clip(xpowers - 1, 0, None)
            newpowersy = np.clip(ypowers - 1, 0, None)

            self.newpolyorder = np.empty((2, 2, 2, 21), dtype=object)
            self.newpolyorder[0][0] = np.stack((newpowersx, ypowers), axis=1)
            self.newpolyorder[0][1] = np.stack((xpowers, newpowersy), axis=1)
            self.newpolyorder[1][0] = np.stack((newpowersx, ypowers), axis=1)
            self.newpolyorder[1][1] = np.stack((xpowers, newpowersy), axis=1)
            self.newpolyorder = np.moveaxis(self.newpolyorder, 3, 2)

            jacob = np.empty((2, 2, 21), dtype=object)
            jacob[0][0] = xpowers * coeff[:, 0]
            jacob[0][1] = ypowers * coeff[:, 0]
            jacob[1][0] = xpowers * coeff[:, 1]
            jacob[1][1] = ypowers * coeff[:, 1]

            mat = np.sum(jacob * np.prod(np.power(self.posOut, self.newpolyorder), axis=3), axis=2)
            mat *= np.pi / 180
        elif method == "raytrace":
            raytrace = RomanRayBundle(
                self.xan, self.yan, 7, self.use_filter, wl=self.wavelength * 0.001, hasE=True
            )
            mat = compute_jacobian(
                raytrace.u,
                dx=raytrace.xyi[0, 1, 0] - raytrace.xyi[0, 0, 0],
                dy=raytrace.xyi[0, 1, 0] - raytrace.xyi[0, 0, 0],
            )[3, 3, :, :]
            # mat has units of inverse mm
        else:
            raise Exception("Invalid method for computing distortion matrix")
        return mat

    def compute_determinant(self):
        """
        Determinant of distortion matrix.

        Returns
        -------
        float
            The determinant of the FPA --> field angle mapping, in mm^-2.

        """

        determinant = (
            self.distortionMatrix[0][0] * self.distortionMatrix[1][1]
            - self.distortionMatrix[1][0] * self.distortionMatrix[0][1]
        )
        return determinant

    def load_pupil_mask(self, use_ray_trace=True):
        """
        Loads the pupil mask.

        Also sets the pupil coordinates.

        Parameters
        ----------
        use_ray_trace : bool, optional
            Ray traced pupil mask? (Only turn off for testing, if stpsf-data is
            available.)

        Returns
        -------
        np.ndarray of float
            The pupil mask.

        """

        jacobian = np.linalg.inv(-self.pupilLength * self.distortionMatrix)
        self.rb = rb = RomanRayBundle(
            self.xan,
            self.yan,
            self.pupilSampling,
            self.use_filter,
            width=self.samplingwidth,
            wl=self.wavelength * 0.001,
            hasE=True,
            jacobian=jacobian,
        )
        if use_ray_trace:
            mask = rb.open
            self.pupil_mask_u = rb.u
            self.pupil_mask_s = rb.s
            self.pupil_mask_xin = rb.xyi
            self.pupil_mask_xout = rb.x_out
        else:
            dirName = "./stpsf-data/WFI/pupils/"
            pupilMaskString = f"SCA{self.scanum}_full_mask.fits.gz"
            file = fits.open(dirName + pupilMaskString)
            mask = file[0].data
        return mask

    def path_diff(self):
        """
        Path difference map.

        Returns
        -------
        np.ndarray of float
            The path difference map; same shape as ``self.u_array()``.
            Units of microns.

        """

        self.urhoPolar = np.sqrt((self.u_array() - self.ucen) ** 2 + (self.v_array() - self.vcen) ** 2)
        self.uthetaPolar = np.arctan2(self.v_array() - self.vcen, self.u_array() - self.ucen)

        infile = files("psfsim.data").joinpath("wim_zernikes_cycle9.csv.gz")  # reads data directory
        mydata = pd.read_csv(infile, sep=",", header=0, compression="gzip")
        # Define mask to desired wavelength and correct X and Y (Need to modify to within bounds):
        snap_wl_0 = np.array(mydata["wavelength"]).flatten()
        snap_wl = snap_wl_0[np.argmin(np.abs(snap_wl_0 - self.wavelength))]
        mask1 = (np.abs(mydata["wavelength"] - snap_wl) < 0.001) & (mydata["sca"] == self.scanum)
        # print(np.where(mask1 == True))
        localx = mydata["local_x"][mask1]
        localy = mydata["local_y"][mask1]
        points = np.stack((localx, localy)).T
        # print(points)
        path_diff = 0
        for i in range(22):
            zIndex = i + 1
            zString = f"Z{zIndex}"
            zernCoeffsToInterpolate = np.asarray(mydata[zString][mask1])
            zernCoeff = altgriddata(points, zernCoeffsToInterpolate, (self.scax, self.scay))
            nZern, mZern = zernike.noll_to_zernike(i + 1)
            # print(">>>", zernike.zernike(nZern, mZern, self.urhoPolar, self.uthetaPolar))
            path_diff += zernCoeff * zernike.zernike(
                nZern, mZern, 2 * self.focalLength * self.urhoPolar, self.uthetaPolar
            )
        return path_diff
