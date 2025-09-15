import numpy as np
from numpy import newaxis as na
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from astropy.io import fits
from scipy.linalg import expm
from scipy.interpolate import griddata
from scipy.fft import ifftn

from filter_detector_properties import filter_detector
from nHgCdTe import nHgCdTe
from opticsPSF import GeometricOptics
import WFI_coordinate_transformations as WFI
from MTF import MTF_SCA

interference_filter = filter_detector(n1=1.5, t1=1./3, n2=1.43, t2=1./3, n3=2.0, t3=1./3,sgn=1)



class PSFObject(object):

    def __init__(self, SCAnum, SCAx, SCAy, wavelength = 0.48, postage_stamp_size=20, npix_boundary=1, use_postage_stamp_size=True):

        '''
        Class denoting a monochromatic PSF -- should have a GeometricOptics object and a wavelength (and possibly others) 
        postage_stamp_size: length of the side of the square postage stamp in pixels
        '''
        self.wavelength = wavelength
        self.npix_boundary = npix_boundary

        #The following sets the ulen of the GeometricOptics object based on the postage_stamp_size if use_postage_stamp_size is True. 
        if use_postage_stamp_size:
            self.ulen = int(self.get_ulen(ps=postage_stamp_size))+1
        else:
            self.ulen = 2048 # default value

        self.Optics = GeometricOptics(SCAnum, SCAx, SCAy, wavelength, self.ulen)
        self.uX, self.uY = np.meshgrid(self.Optics.uX, self.Optics.uY, indexing='ij')
        self.u = np.sqrt(self.uX**2 + self.uY**2)
        self.mask = (self.u <= 1)



        sX = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns
        sY = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns

        self.sX, self.sY = np.meshgrid(sX, sY, indexing='ij')

        self.dsX = self.Optics.wavelength/(self.Optics.umax-self.Optics.umin) # postage stamp pixel size in microns
        self.dsY = self.Optics.wavelength/(self.Optics.umax-self.Optics.umin) # postage stamp pixel size in microns
 

    def get_ulen(self, ps=20):
        '''
        Returns the required ulen for a postage stamp of size ps in pixels
        '''
        pixsize = 10 # pixel size in microns
        smin = -ps*pixsize 
        smax = ps*pixsize # Note that uX and uY have to be fourier duals to twice the size of the postage stamp to avoid aliasing from periodic boundary conditions
        
        ulen = 2*(smax-smin)/self.wavelength
        return ulen

    def get_Optical_PSF(self, normalise=True):
        """
        Returns values of the optical PSF on the SCA surface in the postage stamp surrounding the point (SCAx, SCAy) in the SCA. This function is added for testing purposes and to assess the impact of the interference filter on the PSF and charge diffusion through the HgCdTe layer. Note that the optical PSF includes the effects of diffraction and pupil mask and is normalised to total flux of 1.
        """


        print('pathDifference = ',self.Optics.pathDifference,'\n')
        print(self.Optics.pathDifference.shape)
        print(self.Optics.pupilMask.shape)
        prefactor = self.Optics.pupilMask*self.Optics.determinant*np.exp(2*np.pi/self.wavelength*1j*self.Optics.pathDifference)
        
        x_minus = (-1)**np.array(range(self.Optics.ulen))#used to translate ftt to image center
        ph = np.outer(x_minus, x_minus) #phase required to translate fft to center
        print('ph = ',ph,'\n')
        prefactor *= ph

        E = np.fft.fft2(prefactor)
        self.Optical_PSF = abs(E)**2
        self.Optical_PSF /= np.sum(self.Optical_PSF*self.dsX*self.dsY) # Normalise to total flux of 1



        


    def get_E_in_detector(self,detector_thickness=5, zlen=10, filter=interference_filter):

        ''' Creates self.Ex, self.Ey, self.Ez -- arrays of electric field amplitudes within the detector of thickness tz for self.uX and self.uY. Returns a 3D array of intensity in the postage stamp surrounding the point (SCAx, SCAy) in the SCA and going to a depth of tz. The size of the postage stamp and resolution are determined by ulen.
        Also creates self.Filtered_PSF -- the PSF on the SCA surface after passing through the interference filter, normalised to total flux of 1.
        The interference filter object created earlier is assumed to be the default interference filter.
        '''

        z_array = np.linspace(0, detector_thickness, zlen)
        dZ = z_array[1]-z_array[0]
        ulen = self.Optics.ulen
        uX = self.Optics.uX
        uY = self.Optics.uY
        uX, uY = np.meshgrid(uX, uY, indexing='ij')

        Ex = np.zeros((ulen,ulen),dtype=np.complex128)
        Ey = np.zeros((ulen,ulen),dtype=np.complex128)
        Ez = np.zeros((ulen,ulen),dtype=np.complex128)


        E = filter.Transmitted_E(self.wavelength, uX, uY)
        Ex = E[0]
        Ey = E[1]
        Ez = E[2]




        k0 = 2.*np.pi/self.wavelength # in microns^-1
        n = nHgCdTe(self.wavelength)
        kz = np.zeros_like(uX, dtype=np.complex128)
        kz[mask] = k0*np.sqrt(n**2 - uX[:,na]**2 - uY[na,:]**2) # in microns^-1
        mask = (uX**2 + uY**2) <= 1.0
        kz[mask & (kz.imag < 0)] = -kz[mask & (kz.imag < 0)]
        for index_z in range(1, zlen):
            z = z_array[index_z]
            attenuation = np.exp(1j*kz*z)
            Ex[:,:,index_z] = Ex[:,:,0]*attenuation
            Ey[:,:,index_z] = Ey[:,:,0]*attenuation
            Ez[:,:,index_z] = Ez[:,:,0]*attenuation
            print(f'Completed z index ', index_z, ' at depth z = ', z, ' microns')
                
        prefactor = self.Optics.pupilMask*self.Optics.determinant*np.exp(2*np.pi/self.wavelength*1j*self.Optics.pathDifference)

        x_minus = (-1)**np.array(range(ulen))#used to translate ftt to image center
        ph = np.outer(x_minus, x_minus) #phase required to translate fft to center
        
        prefactor *= ph

        Ex *= prefactor[:,:,na]
        Ey *= prefactor[:,:, na]
        Ez *= prefactor[:,:, na]

        

        Ex_postage_stamp = np.fft.fft2(Ex, axes=(0,1))
        Ey_postage_stamp = np.fft.fft2(Ey, axes=(0,1))
        Ez_postage_stamp = np.fft.fft2(Ez, axes=(0,1))
        
        Intensity = (abs(Ex_postage_stamp)**2) + (abs(Ey_postage_stamp)**2) + (abs(Ez_postage_stamp)**2)

        self.Intensity = np.sum(Intensity*dZ, axis=2)

        self.Filtered_PSF = self.Intensity[:,:,0]/np.sum(self.Intensity[:,:,0]*self.dsX*self.dsY) # Filtered PSF normalise to total flux of 1 (introduced only for testing purposes)


        return


    def get_detector_image(self):
        """
        Returns the 4088x4088 detector image as a 2D array of intensity values.
        """

        if not hasattr(self, 'Intensity'):
            self.get_E_in_detector()


        # Compute the detector image by summing the contributions from all points in the postage stamp
        detector_image = np.zeros((4088, 4088), dtype=np.float64)

        XAnalysis, YAnalysis = WFI.fromSCAtoAnalysis(self.Optics.scaNum, self.Optics.scaX, self.Optics.scaY) #Center of the PSF in Analysis coordinates


        for index_sx in range(self.Optics.ulen):
            for index_sy in range(self.Optics.ulen):
                sx = self.sX[index_sx]
                sy = self.sY[index_sy]

                detector_image += MTF_SCA(XAnalysis+sx, YAnalysis+sy, npix_boundary=self.npix_boundary) * self.Intensity[index_sx, index_sy] * self.dsX * self.dsY

        self.detector_image = detector_image


        
    



