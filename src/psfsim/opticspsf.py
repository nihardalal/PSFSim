"""Optics objects."""

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import griddata

from . import zernike
from .romantrace import RomanRayBundle
from .wfi_coordinate_transformations import fromFPAtoAngle, fromSCAToFPA


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
    SCAnum : int
        The SCA number.
    SCAx, SCAy : float
        The pixel positions on the SCA (in mm, FPA coordinates relative to the SCA center).
    wavelength : float, optional
        The vacuum wavelength in microns.
    ulen : int, optional
        The size of array for pupil sampling.
    ray_trace : bool, optional
        Whether to use ray tracing.
    pixelsampling : float, optional
        Desired FFT-based output pixel sampling in microns.

    """

    def __init__(self, SCAnum, SCAx, SCAy, wavelength=0.48, ulen=2048, ray_trace=False, pixelsampling=1.00):
        # sca position in mm
        # wavelength in micrometers

        self.wavelength = wavelength
        self.dsX = pixelsampling  # pixel spacing in microns
        self.pupilLength = 2400 * 8  # in mm
        self.focalLength = 8  # m
        self.samplingwidth = (self.wavelength / self.dsX) * self.pupilLength  # in mm for raytrace

        self.scaNum = SCAnum
        self.scaX = SCAx
        self.scaY = SCAy
        # print(self.scaX, self.scaY)

        self.xout, self.yout = fromSCAToFPA(self.scaNum, self.scaX, self.scaY)
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
        self.xan, self.yan = fromFPAtoAngle(self.posOut, wavelength=self.wavelength)

        self.usefilter = "H"

        # Compute Distortion Matrix and dterminant
        self.distortionMatrix = self.computeDistortionMatrix(method="raytrace")
        self.determinant = self.computeDeterminant()

        # Obtain pupil Mask
        self.pupilMask = self.loadPupilMask(use_ray_trace=ray_trace)

        # Load pupil mask from raytrace - more accurate
        self.uArray = self.pupilMaskU[:, :, 0]
        self.vArray = self.pupilMaskU[:, :, 1]
        self.umin = np.min(self.uArray)
        self.umax = np.max(self.uArray)
        self.vmin = np.min(self.vArray)
        self.vmax = np.max(self.vArray)
        self.ucen = 0.5 * (self.umin + self.umax)
        self.vcen = 0.5 * (self.vmin + self.vmax)
        # Get path difference map
        self.pathDifference = self.pathDiff()

        # self.integrand = self.pupilMask*self.determinant\
        # *expm(2*np.pi/self.wavelength*1j*self.pathDifference)
        # self.x_minus = (-1)**np.array(range(ulen))#used to translate ftt to image center
        # self.ph = np.outer(self.x_minus, self.x_minus) #phase required to translate fft to center
        # self.eArray = np.fft.fft2(self.integrand*self.ph)
        # self.magEArray = abs(self.eArray)

    # Compute distortion matrix here!
    # Should return [[d(FPAx)/d(xan), d(FPAx)/d(yan)], [d(FPAy)/d(xan), d(FPAy)/d(yan)]]
    def computeDistortionMatrix(self, method="raytrace"):
        """Computes the distortion matrix"""

        if method == "poly":
            # Load in polynomial fits to Jacobian
            jacobian_fit_file_name = "./data/FPAtoAnglePoly.npy"
            self.jacobian_fits = np.load(jacobian_fit_file_name)
            self.wavindex = np.where(self.jacobian_fits["wavelength"] == self.wavelength)
            self.coeff = self.jacobian_fits["coefficients"][self.wavindex]
            self.exponents = self.jacobian_fits["exponents"][self.wavindex]
            xpowers = self.exponents[:, :, 0]
            ypowers = self.exponents[:, :, 1]
            newpowersx = np.clip(xpowers - 1, 0, None)
            newpowersy = np.clip(ypowers - 1, 0, None)

            self.newpolyorder = np.empty((2, 2, 2, 21), dtype=object)
            self.newpolyorder[0][0] = np.stack((newpowersx, ypowers), axis=1)
            self.newpolyorder[0][1] = np.stack((xpowers, newpowersy), axis=1)
            self.newpolyorder[1][0] = np.stack((newpowersx, ypowers), axis=1)
            self.newpolyorder[1][1] = np.stack((xpowers, newpowersy), axis=1)
            self.newpolyorder = np.moveaxis(self.newpolyorder, 3, 2)

            jacob = np.empty((2, 2, 21), dtype=object)
            jacob[0][0] = xpowers * self.coeff[:, :, 0]
            jacob[0][1] = ypowers * self.coeff[:, :, 0]
            jacob[1][0] = xpowers * self.coeff[:, :, 1]
            jacob[1][1] = ypowers * self.coeff[:, :, 1]

            mat = np.sum(jacob * np.prod(np.power(self.posOut, self.newpolyorder), axis=3), axis=2)
            mat *= np.pi / 180
        elif method == "raytrace":
            raytrace = RomanRayBundle(self.xan, self.yan, 8, "H", wl=self.wavelength * 0.001, hasE=True)
            mat = compute_jacobian(
                raytrace.u,
                dx=raytrace.xyi[0, 1, 0] - raytrace.xyi[0, 0, 0],
                dy=raytrace.xyi[0, 1, 0] - raytrace.xyi[0, 0, 0],
            )[3, 3, :, :]
            # mat has units of inverse mm
        else:
            raise Exception("Invalid method for computing distortion matrix")
        return mat

    def computeDeterminant(self):
        """Determinant of distortion matrix"""

        determinant = (
            self.distortionMatrix[0][0] * self.distortionMatrix[1][1]
            - self.distortionMatrix[1][0] * self.distortionMatrix[0][1]
        )
        return determinant

    def loadPupilMask(self, use_ray_trace=False):
        """Loads the pupil mask."""

        if use_ray_trace:
            jacobian = np.linalg.inv(-self.pupilLength * self.distortionMatrix)
            rb = RomanRayBundle(
                self.xan,
                self.yan,
                self.pupilSampling,
                self.usefilter,
                width=self.samplingwidth,
                wl=self.wavelength * 0.001,
                hasE=True,
                jacobian=jacobian,
            )
            mask = rb.open
            self.pupilMaskU = rb.u
            self.pupilMaskS = rb.s
            self.pupilMaskXin = rb.xyi
            self.pupilMaskXout = rb.x_out
        else:
            dirName = "./stpsf-data/WFI/pupils/"
            pupilMaskString = f"SCA{self.scaNum}_full_mask.fits.gz"
            file = fits.open(dirName + pupilMaskString)
            mask = file[0].data
        return mask

    def pathDiff(self):
        """Path difference map"""

        self.urhoPolar = np.sqrt((self.uArray - self.ucen) ** 2 + (self.vArray - self.vcen) ** 2)
        self.uthetaPolar = np.arctan2(self.vArray - self.vcen, self.uArray - self.ucen)

        mydata = pd.read_csv("stpsf-data/WFI/wim_zernikes_cycle9.csv", sep=",", header=0)
        # Define mask to desired wavelength and correct X and Y (Need to modify to within bounds):
        mask1 = (mydata["wavelength"] == self.wavelength) & (mydata["sca"] == self.scaNum)
        # print(np.where(mask1 == True))
        localx = mydata["local_x"][mask1]
        localy = mydata["local_y"][mask1]
        points = np.stack((localx, localy)).T
        # print(points)
        pathDiff = 0
        for i in range(22):
            zIndex = i + 1
            zString = f"Z{zIndex}"
            zernCoeffsToInterpolate = mydata[zString][mask1]
            # print(zernCoeffsToInterpolate)
            zernCoeff = griddata(points, zernCoeffsToInterpolate, (self.scaX, self.scaY), method="linear")
            # print(zernCoeff)
            nZern, mZern = zernike.noll_to_zernike(i + 1)
            # print(zernike.zernike(nZern, mZern, self.urhoPolar, self.uthetaPolar))
            pathDiff += zernCoeff * zernike.zernike(
                nZern, mZern, 2 * self.focalLength * self.urhoPolar, self.uthetaPolar
            )
        return pathDiff
