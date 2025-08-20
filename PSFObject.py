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


interference_filter = filter_detector(n1=1.5, t1=0.3, n2=1.5, t2=0.3, n3=1.5, t3=0.3,sgn=1)



class PSFObject(object):

    def __init__(self, SCAnum, SCAx, SCAy, wavelength = 0.48, ulen = 2048):

        '''
        Class denoting a monochromatic PSF -- should have a GeometricOptics object and a wavelength (and possibly others) 
        '''

        self.Optics = GeometricOptics(SCAnum, SCAx, SCAy, wavelength, ulen)
        self.wavelength = wavelength
            
        # Add other attributes later -- like a RayBundle instance? 

    def get_E_in_detector(self,z=0):

        ''' Creates self.Ex, self.Ey, self.Ex -- arrays of electric field amplitudes within the detector at depth zp for self.uX and self.uY. Ideally should create a 3D array at points (xp, yp, zp)
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

 
    def drawImage(self):
        # Add later
        pass



