import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from astropy.io import fits
from scipy.linalg import expm
from scipy.interpolate import griddata
from scipy.fft import ifftn

from filter_detector_properties import filter_detector
from opticsPSF import GeometricOptics
import WFI_coordinate_transformations as WFI
from MTF import MTF_SCA

interference_filter = filter_detector(n1=1.5, t1=0.3, n2=1.5, t2=0.3, n3=1.5, t3=0.3,sgn=1)



class PSFObject(object):

    def __init__(self, SCAnum, SCAx, SCAy, wavelength = 0.48, ulen = 2048, npix_boundary=1):

        '''
        Class denoting a monochromatic PSF -- should have a GeometricOptics object and a wavelength (and possibly others) 
        '''

        self.Optics = GeometricOptics(SCAnum, SCAx, SCAy, wavelength, ulen)
        self.wavelength = wavelength
        self.npix_boundary = npix_boundary

        self.sX = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns
        self.sY = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns

        
        


    def get_E_in_detector(self,z=0):

        ''' Creates self.Ex, self.Ey, self.Ex -- arrays of electric field amplitudes within the detector at depth zp for self.uX and self.uY. Returns a 2D array of intensity in the postage stamp surrounding the point (SCAx, SCAy) in the SCA. The size of the postage stamp and resolution are determined by ulen.
        '''
        ulen = self.Optics.ulen
        uX = self.Optics.uX
        uY = self.Optics.uY

        Ex = np.zeros((ulen,ulen),dtype=np.complex128)
        Ey = np.zeros((ulen,ulen),dtype=np.complex128)
        Ez = np.zeros((ulen,ulen),dtype=np.complex128)

        # May need to to use itertools instead of nested for loops 
        for index_ux in range(ulen):
            for index_uy in range(ulen):

                ux = uX[index_ux]
                uy = uY[index_uy]

                E = interference_filter.Transmitted_E(self.wavelength, ux, uy, zp=z)
                
                Ex[index_ux, index_uy] = E[0]
                Ey[index_ux, index_uy] = E[1]
                Ez[index_ux, index_uy] = E[2]
                
        prefactor = self.Optics.pupilMask*self.Optics.determinant*expm(2*np.pi/self.wavelength*1j*self.Optics.pathDifference)

        x_minus = (-1)**np.array(range(ulen))#used to translate ftt to image center
        ph = np.outer(self.x_minus, self.x_minus) #phase required to translate fft to center
        
        prefactor *= ph

        Ex *= prefactor
        Ey *= prefactor
        Ez *= prefactor

        

        self.Ex = np.fft.fft2(Ex)
        self.Ey = np.fft.fft2(Ey)
        self.Ez = np.fft.fft2(Ez)
        
        self.Intensity = (abs(self.Ex)**2) + (abs(self.Ey)**2) + (abs(self.Ez)**2)
        return

    def get_MTF_SCA(self, npix_boundary=1):

        """
        Returns a 2D array of MTF_SCA outputs for the points in the postage stamp with postage stamp coordinates (sX, sY) in microns.
        npix_boundary: number of pixels in the boundary layer where reflection boundary conditions are applied
        """
        nside = 4088
        MTF_SCA_array = np.zeros((self.Optics.ulen, self.Optics.ulen, nside, nside), dtype=np.complex128)

        XAnalysis, YAnalysis = WFI.SCAtoAnalysis(self.Optics.scaNum, self.Optics.scaX, self.Optics.scaY) #Center of the PSF in Analysis coordinates

        for index_sx in range(self.Optics.ulen):
            for index_sy in range(self.Optics.ulen):

                sx = self.sX[index_sx]
                sy = self.sY[index_sy]

                MTF_SCA_array[index_sx, index_sy, :, :] = MTF_SCA(XAnalysis+sx, YAnalysis+sy, npix_boundary=npix_boundary)

        self.MTF_SCA_array = MTF_SCA_array

    def get_detector_image(self):
        """
        Returns the 4088x4088 detector image as a 2D array of intensity values.
        """

        if not hasattr(self, 'Intensity'):
            self.get_E_in_detector()

        if not hasattr(self, 'MTF_SCA_array'):
            self.get_MTF_SCA(npix_boundary=self.npix_boundary)

        # Compute the detector image by summing the contributions from all points in the postage stamp
        detector_image = np.zeros((4088, 4088), dtype=np.float64)

        dsX = self.Optics.wavelength / (self.Optics.umax - self.Optics.umin)
        dsY = self.Optics.wavelength / (self.Optics.umax - self.Optics.umin)

        for index_sx in range(self.Optics.ulen):
            for index_sy in range(self.Optics.ulen):
                detector_image += self.MTF_SCA_array[index_sx, index_sy, :, :] * self.Intensity[index_sx, index_sy] * dsX * dsY

        self.detector_image = detector_image


        
    



