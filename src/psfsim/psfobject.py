# from numba import njit, prange
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy import newaxis as na
from scipy.fft import ifft2
from scipy.signal import fftconvolve

from . import wfi_coordinate_transformations as wfi
from .filter_detector_properties import FilterDetector
from .mtf_diffusion import MTF_image, MTF_SCA_postage_stamp
from .opticspsf import GeometricOptics
from .zernike import noll_to_zernike, zernike

c = 3.0e8  # speed of light in m/s
epsilon_0 = 8.8541878188e-12  # permittivity of free space in F/m


def parallel_MTF_image(args):
    """Wrapper for MTF_image"""

    xd, yd, imageX, imageY, Intensity_integrated, npix_boundary = args
    return MTF_image(xd, yd, imageX, imageY, Intensity_integrated, npix_boundary)


interference_filter = FilterDetector([1.5, 1.43, 2.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1)


class PSFObject:
    """
    Monochromatic PSF object class.

    Parameters
    ----------
    scanum : int
        The SCA number.
    scax, scay : float
        The pixel positions on the SCA (in mm, FPA coordinates relative to the SCA center).
    wavelength : float, optional
        The vacuum wavelength in microns.
    postage_stamp_size : int, optional
        The length of the side of the square postage stamp in native pixels.
    ovsamp : int, optional
        The oversampling factor for the PSF (number of samples per native pixel).
    use_filter : str, optional
        The filter configuration to use (1-character code).
    npix_boundary : int, optional
        ?
    use_postage_stamp_size : int, optional
        Force pupil postage stamp size instead of internal calculation.
    ray_trace : bool, optional
        Whether to use ray tracing. (Only turn off for testing.)
    add_focus : variable
        Parameter for adding focus.


    Attributes
    ----------
    wavelength : float
        The wavelength
    interference_filter : psfsim.filter_detector_properties.FilterDetector
        The interference filter object.
    ulen : int
         The length of the FFTs.
    optics : psfsim.opticspsf.GeometricOptics
         The Geometric Optics object.

    Methods
    -------
    __init__
        Constructor.
    get_optical_psf
        Gets the optical PSF (no detector effects).

    """

    def __init__(
        self,
        scanum,
        scax,
        scay,
        wavelength=0.48,
        postage_stamp_size=31,
        ovsamp=10,
        use_filter="H",
        npix_boundary=1,
        use_postage_stamp_size=None,
        ray_trace=True,
        add_focus=None,
    ):
        self.wavelength = wavelength
        self.npix_boundary = npix_boundary

        self.interference_filter = FilterDetector([1.5, 1.43, 2.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1)
        self.postage_stamp_size = postage_stamp_size

        # The following sets the ulen of the GeometricOptics object based on the postage_stamp_size if
        # use_postage_stamp_size is True.
        self.ulen = 2048  # default value
        if use_postage_stamp_size:
            self.ulen = use_postage_stamp_size

        self.optics = GeometricOptics(
            scanum,
            scax,
            scay,
            wavelength=wavelength,
            use_filter=use_filter,
            ulen=self.ulen,
            ray_trace=ray_trace,
            pixelsampling=10.0 / ovsamp,
        )
        self.ux, self.uy = (
            self.optics.u_array(),
            self.optics.v_array(),
        )  # np.meshgrid(self.Optics.uX, self.Optics.uY, indexing='ij')
        self.u = np.sqrt(self.ux**2 + self.uy**2)
        self.mask = self.u <= 1

        # sX = (self.wavelength / (self.optics.umax - self.optics.umin)) * (
        #     -(self.optics.ulen / 2.0) + np.array(range(self.optics.ulen))
        # )  # postage stamp coordinates along the FPA axes in microns
        # sY = (self.wavelength / (self.optics.umax - self.optics.umin)) * (
        #     -(self.optics.ulen / 2.0) + np.array(range(self.optics.ulen))
        # )  # postage stamp coordinates along the FPA axes in microns

        # self.sX, self.sY = np.meshgrid(sX, sY, indexing="ij")

        # self.dsX = self.optics.wavelength / (
        #     self.optics.umax - self.optics.umin
        # )  # postage stamp pixel size in microns
        # self.dsY = self.optics.wavelength / (
        #     self.optics.umax - self.optics.umin
        # )  # postage stamp pixel size in microns

        self.dx = self.optics.wavelength / np.abs(self.ulen * self.optics.du)

        if add_focus is not None:
            nZern, mZern = noll_to_zernike(4)
            self.optics.pathDifference += add_focus * zernike(
                nZern, mZern, 2 * self.optics.focalLength * self.optics.urhoPolar, self.optics.uthetaPolar
            )

        prefactor = (
            self.optics.pupil_mask
            / self.optics.determinant
            * np.exp(2 * np.pi / self.wavelength * 1j * self.optics.path_difference)
        )

        x_minus = (-1) ** np.array(range(self.ulen))  # used to translate ftt to image center
        ph = np.outer(x_minus, x_minus)  # phase required to translate fft to center

        self.prefactor = prefactor * ph

        # self.MTF_array = diffusion_green(self.sX, self.sY)

    #    def get_ulen(self, ps=20):
    #       # Returns the required ulen for a postage stamp of size ps in pixels.
    #
    #        pixsize = wfi_data.pix  # pixel size in microns
    #        smin = -ps * pixsize
    #        smax = (
    #            ps * pixsize
    #        )  # Note that uX and uY have to be fourier duals to twice the size of the postage stamp to
    #        # avoid aliasing from periodic boundary conditions
    #
    #        ulen = 2 * (smax - smin) / self.wavelength
    #        return ulen

    def get_optical_psf(self, normalise=True):
        """
        Gets the optical PSF (no detector effects).

        Returns values of the
        optical PSF on the SCA surface in the postage stamp surrounding the point (SCAx, SCAy) in the
        SCA. This function is added for testing purposes and to assess the impact of the interference
        filter on the PSF and charge diffusion through the HgCdTe layer. Note that the optical PSF
        includes the effects of diffraction and pupil mask and is normalised to total flux of 1.

        Parameters
        ----------
        normalise : bool, optional
            Currently has no effect.
        """

        # prefactor = \
        # self.optics.pupilMask*self.optics.determinant*np.exp(2*np.pi/self.wavelength*1j\
        # *self.optics.pathDifference)

        # x_minus = (-1)**np.array(range(self.optics.ulen))#used to translate ftt to image center
        # ph = np.outer(x_minus, x_minus) #phase required to translate fft to center
        # prefactor *= ph
        # start_time = time.time()
        # current_time = time.time()
        # old version by Charuhas below
        # E_local = np.zeros(self.ux.shape + (3,), dtype=np.complex128)

        # E_local[self.mask, 0] = A_TE * np.ones_like(self.ux[self.mask])
        # E_local[self.mask, 1] = -np.sqrt(1 - self.u[self.mask] ** 2) * A_TM
        # E_local[self.mask, 2] = self.u[self.mask] * A_TM

        # local_to_FPA = local_to_fpa_rotation(self.ux, self.uy, sgn=1)

        # E_FPA_x = np.zeros_like(self.ux, dtype=np.complex128)
        # E_FPA_y = np.zeros_like(self.ux, dtype=np.complex128)
        # E_FPA_z = np.zeros_like(self.ux, dtype=np.complex128)

        # E_FPA_x[self.mask] = np.sum(local_to_FPA[self.mask, 0, :] * E_local[self.mask, :], axis=-1)
        # E_FPA_y[self.mask] = np.sum(local_to_FPA[self.mask, 1, :] * E_local[self.mask, :], axis=-1)
        # E_FPA_z[self.mask] = np.sum(local_to_FPA[self.mask, 2, :] * E_local[self.mask, :], axis=-1)
        # end_time = time.time()
        # print("Time taken to get E field in FPA coordinates = ", end_time - current_time, "\n")
        # current_time = time.time()
        # E_FPA_x *= self.prefactor
        # E_FPA_y *= self.prefactor
        # E_FPA_z *= self.prefactor

        # Ex = ifft2(E_FPA_x)
        # Ey = ifft2(E_FPA_y)
        # Ez = ifft2(E_FPA_z)

        # Ex = np.fft.ifft2(E_FPA_x, axes=(
        # print("Time taken to do ifft = ", time.time() - current_time, "\n")
        # current_time = time.time()

        # self.Optical_PSF = abs(Ex) ** 2 + abs(Ey) ** 2 + abs(Ez) ** 2
        # print("Time taken to compute Optical PSF by squaring the E field = ",
        # time.time()-current_time, "\n")
        # self.Optical_PSF /= np.sum(self.Optical_PSF*self.dsX*self.dsY) # Normalise to total flux of 1
        # self.Optical_PSF *= np.sum(self.dsX*self.dsY)

        # New changes by Nihar here, please check before removing this comment
        # Goal is to get polarization consistent with raytrace

        E_FPA_x_polarized = self.optics.rb.E[:, :, 0, 1:4]
        E_FPA_y_polarized = self.optics.rb.E[:, :, 1, 1:4]

        E_FPA_x_polarized = self.prefactor[:, :, np.newaxis] * E_FPA_x_polarized
        E_FPA_y_polarized = self.prefactor[:, :, np.newaxis] * E_FPA_y_polarized

        r = np.array(
            [self.ux, self.uy, np.sqrt(1 - self.u**2)]
        )  # define a vector along propagation direction
        r = r.reshape(self.ux.shape[0], self.ux.shape[1], 3)  # reshape to be compatible with E
        cB_FPA_x_polarized = np.cross(r, E_FPA_x_polarized)
        cB_FPA_y_polarized = np.cross(r, E_FPA_y_polarized)

        # Need to add normalization
        E_x_polarized = ifft2(E_FPA_x_polarized, axes=(0, 1))  # use first two axes for fft
        E_y_polarized = ifft2(E_FPA_y_polarized, axes=(0, 1))
        cB_x_polarized = ifft2(cB_FPA_x_polarized, axes=(0, 1))
        cB_y_polarized = ifft2(cB_FPA_y_polarized, axes=(0, 1))

        # Unsure about the abs here, but leaving it in for now...
        self.x_polarized_psf = np.abs(
            0.5
            * epsilon_0
            * c
            * (
                E_x_polarized[:, :, 0] * cB_x_polarized[:, :, 1]
                - E_x_polarized[:, :, 1] * cB_x_polarized[:, :, 0]
            )
        )
        self.y_polarized_psf = np.abs(
            0.5
            * epsilon_0
            * c
            * (
                E_y_polarized[:, :, 0] * cB_y_polarized[:, :, 1]
                - E_y_polarized[:, :, 1] * cB_y_polarized[:, :, 0]
            )
        )
        self.Optical_PSF = 0.5 * (self.x_polarized_psf + self.y_polarized_psf)

        return

    def get_E_in_detector(
        self, filter=interference_filter, detector_thickness=2, zlen=20, nworkers=8, A_TE=1.0e10, A_TM=1.0e10
    ):
        """
        Creates self.Ex, self.Ey, self.Ez -- arrays of electric field amplitudes within the
        detector of thickness tz for self.uX and self.uY. Returns a 3D array of intensity in the
        postage stamp surrounding the point (SCAx, SCAy) in the SCA and going to a depth of tz. The
        size of the postage stamp and resolution are determined by ulen. Also creates
        self.Filtered_PSF -- the PSF on the SCA surface after passing through the interference filter,
        normalised to total flux of 1. The interference filter object created earlier is assumed to be
        the default interference filter.
        """

        start_time = time.time()
        current_time = time.time()
        # print('Starting get_E_in_detector at time = ',current_time,'\n')
        # first_time = time.time()
        z_array = np.linspace(0, detector_thickness, zlen)
        # dZ = z_array[1] - z_array[0]
        # ulen = self.optics.ulen

        uX = self.optics.u_array()
        uY = self.optics.v_array()
        # uX, uY = np.meshgrid(uX, uY, indexing='ij')
        # uX, uY = np.meshgrid(self.uX, self.uY, indexing='ij')

        E = filter.Transmitted_E(self.wavelength, uX, uY, z_array, A_TE=A_TE, A_TM=A_TM)
        Ex = E[0]
        Ey = E[1]
        Ez = E[2]

        end_time = time.time()
        print("Time taken to get transmitted E field through filter = ", end_time - current_time, "\n")
        current_time = time.time()

        Ex *= self.prefactor[:, :, na]
        Ey *= self.prefactor[:, :, na]
        Ez *= self.prefactor[:, :, na]

        end_time = time.time()
        print("Time taken to multiply by prefactor = ", end_time - current_time, "\n")
        current_time = time.time()

        Ex_postage_stamp = ifft2(Ex, axes=(0, 1), workers=nworkers)
        Ey_postage_stamp = ifft2(Ey, axes=(0, 1), workers=nworkers)
        Ez_postage_stamp = ifft2(Ez, axes=(0, 1), workers=nworkers)
        # Ex_postage_stamp = np.fft.ifft2(Ex, axes=(0,1))
        # Ey_postage_stamp = np.fft.ifft2(Ey, axes=(0,1))
        # Ez_postage_stamp = np.fft.ifft2(Ez, axes=(0,1))
        end_time = time.time()
        print("Time taken to do ifft = ", end_time - current_time, "\n")

        current_time = time.time()
        Intensity = (abs(Ex_postage_stamp) ** 2) + (abs(Ey_postage_stamp) ** 2) + (abs(Ez_postage_stamp) ** 2)

        print("Time taken to compute Intensity by squaring the E field = ", time.time() - current_time, "\n")
        current_time = time.time()

        self.Filtered_PSF = Intensity[:, :, 0]  # /np.sum(Intensity[:,:,0]*self.dsX*self.dsY)
        # Filtered PSF normalise to total flux of 1 (introduced only for testing purposes)
        # self.Filtered_PSF *= np.sum(self.dsX*self.dsY)
        end_time = time.time()
        print("Time taken to calculate Filtered PSF = ", end_time - current_time, "\n")
        current_time = time.time()
        self.Intensity = Intensity
        self.Intensity_integrated = np.trapz(Intensity, x=z_array, axis=2)
        end_time = time.time()
        print("Time taken to integrate over depth = ", time.time() - current_time, "\n")

        print("Total time taken for get_E_in_detector = ", time.time() - start_time, "\n")

        return

    def get_detector_image3(self):
        """
        Returns the postage_stamp_size x postage_stamp_size detector image as a 2D array of intensity values.
        """

        # if not hasattr(self, 'Intensity'):
        #    self.get_E_in_detector()

        self.detector_image3 = fftconvolve(self.Intensity_integrated, self.MTF_array, mode="same")

        # XAnalysis, YAnalysis = wfi.fromSCAtoAnalysis(self.optics.scaNum, self.optics.scaX,
        # self.optics.scaY) #Center of the PSF in Analysis coordinates

        # imageX = XAnalysis + self.sX[:,0] # Note that self.sX and self.sY are in microns whereas
        # Analysis coordinates and MTF are in mm

        # imageY = YAnalysis + self.sY[0,:]

        # MTF_array = np.zeros_like(self.sX, dtype=np.float64)

    def get_detector_image2(self):
        """
        Returns the postage_stamp_size x postage_stamp_size detector image as a 2D array of intensity values.
        """

        # if not hasattr(self, 'Intensity'):
        #    self.get_E_in_detector()

        pix = 1.0
        # ps = self.ulen / pix

        # Compute the detector image by summing the contributions from all points in the postage stamp
        # detector_image = np.zeros((, 4088, self.optics.ulen, self.optics.ulen), dtype=np.float64)

        XAnalysis, YAnalysis = wfi.fromSCAtoAnalysis(
            self.optics.scaNum, self.optics.scaX, self.optics.scaY
        )  # Center of the PSF in Analysis coordinates

        imageX = XAnalysis + self.sX[:, 0]  # Note that self.sX and self.sY and
        imageY = YAnalysis + self.sY[0, :]

        Xd = np.floor(XAnalysis // pix) * pix
        Yd = np.floor(YAnalysis // pix) * pix
        xd_array = (
            Xd
            - (np.floor((self.postage_stamp_size - 1) / 2) * pix)
            + pix * np.arange(int(self.postage_stamp_size))
        )
        yd_array = (
            Yd
            - (np.floor((self.postage_stamp_size - 1) / 2) * pix)
            + pix * np.arange(int(self.postage_stamp_size))
        )

        xD, yD = np.meshgrid(xd_array, yd_array, indexing="ij")

        result = MTF_SCA_postage_stamp(imageX, imageY, xD, yD, self.Intensity_integrated, self.npix_boundary)
        self.detector_image2 = result

    def get_detector_image(self, nworkers=8, chunk_size=1):
        """
        Returns the postage_stamp_size x postage_stamp_size detector image as a 2D array of intensity values.
        """

        # if not hasattr(self, 'Intensity'):
        #    self.get_E_in_detector()

        pix = 10
        # Compute the detector image by summing the contributions from all points in the postage stamp
        # detector_image = np.zeros((, 4088, self.optics.ulen, self.optics.ulen), dtype=np.float64)

        XAnalysis, YAnalysis = wfi.fromSCAtoAnalysis(
            self.optics.scaNum, self.optics.scaX, self.optics.scaY
        )  # Center of the PSF in Analysis coordinates

        imageX = (
            XAnalysis + self.sX
        )  # Note that self.sX and self.sY are in microns whereas Analysis coordinates and MTF are in mm
        imageY = YAnalysis + self.sY
        self.imageX = imageX
        self.imageY = imageY

        Xd = np.floor(XAnalysis // pix) * pix
        Yd = np.floor(YAnalysis // pix) * pix
        xd_array = (
            Xd
            - (np.floor((self.postage_stamp_size - 1) / 2) * pix)
            + pix * np.arange(int(self.postage_stamp_size))
        )
        yd_array = (
            Yd
            - (np.floor((self.postage_stamp_size - 1) / 2) * pix)
            + pix * np.arange(int(self.postage_stamp_size))
        )

        xD, yD = np.meshgrid(xd_array, yd_array, indexing="ij")
        mask = (np.maximum(np.abs(xD), np.abs(yD)) <= 20440).astype(
            np.float64
        )  # Mask to zero out values outside the SCA
        shape = (int(self.postage_stamp_size), int(self.postage_stamp_size))

        detector_image = np.zeros(shape, dtype=np.float64)

        tasks = [
            (
                xd_array[index_xd],
                yd_array[index_yd],
                imageX,
                imageY,
                self.Intensity_integrated,
                self.npix_boundary,
            )
            for index_xd in range(self.postage_stamp_size)
            for index_yd in range(self.postage_stamp_size)
        ]

        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            results = list(executor.map(parallel_MTF_image, tasks, chunksize=chunk_size))

        detector_image = np.array(results).reshape(shape)
        # Mask out values outside the SCA
        detector_image *= mask
        self.detector_image = detector_image
