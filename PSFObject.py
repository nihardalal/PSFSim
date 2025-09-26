import numpy as np
from numpy import newaxis as na
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from astropy.io import fits
from scipy.linalg import expm
from scipy.interpolate import griddata
from scipy.fft import ifftn, ifft2
from concurrent.futures import ProcessPoolExecutor
from filter_detector_properties import filter_detector
from nHgCdTe import nHgCdTe
from opticsPSF import GeometricOptics
import WFI_coordinate_transformations as WFI
from MTF import *
import time 
#from numba import njit, prange
import os

def parallel_MTF_image(args):
    xd, yd, imageX, imageY, Intensity_integrated, npix_boundary = args
    return MTF_image(xd, yd, imageX, imageY, Intensity_integrated, npix_boundary)

interference_filter = filter_detector(n1=1.5, t1=1./3, n2=1.43, t2=1./3, n3=2.0, t3=1./3,sgn=1)

class PSFObject(object):

    def __init__(self, SCAnum, SCAx, SCAy, wavelength = 0.48, postage_stamp_size=31, npix_boundary=1, use_postage_stamp_size=True, ray_trace=False):

        '''
        Class denoting a monochromatic PSF -- should have a GeometricOptics object and a wavelength (and possibly others) 
        postage_stamp_size: length of the side of the square postage stamp in pixels
        '''
        self.wavelength = wavelength
        self.npix_boundary = npix_boundary

        self.interference_filter = filter_detector(n1=1.5, t1=1./3, n2=1.43, t2=1./3, n3=2.0, t3=1./3,sgn=1)
        self.postage_stamp_size = postage_stamp_size
        #The following sets the ulen of the GeometricOptics object based on the postage_stamp_size if use_postage_stamp_size is True. 
        if use_postage_stamp_size:
            self.ulen = int(self.get_ulen(ps=postage_stamp_size))+1
        else:
            self.ulen = 2048 # default value

        self.Optics = GeometricOptics(SCAnum, SCAx, SCAy, wavelength, self.ulen, ray_trace=ray_trace)
        self.uX, self.uY = np.meshgrid(self.Optics.uX, self.Optics.uY, indexing='ij')
        self.u = np.sqrt(self.uX**2 + self.uY**2)
        self.mask = (self.u <= 1)



        sX = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns
        sY = (self.wavelength/(self.Optics.umax-self.Optics.umin))*(-(self.Optics.ulen/2.) + np.array(range(self.Optics.ulen))) # postage stamp coordinates along the FPA axes in microns

        self.sX, self.sY = np.meshgrid(sX, sY, indexing='ij')

        self.dsX = self.Optics.wavelength/(self.Optics.umax-self.Optics.umin) # postage stamp pixel size in microns
        self.dsY = self.Optics.wavelength/(self.Optics.umax-self.Optics.umin) # postage stamp pixel size in microns
 
        prefactor = self.Optics.pupilMask*self.Optics.determinant*np.exp(2*np.pi/self.wavelength*1j*self.Optics.pathDifference)

        x_minus = (-1)**np.array(range(self.ulen))#used to translate ftt to image center
        ph = np.outer(x_minus, x_minus) #phase required to translate fft to center
        
        self.prefactor = prefactor*ph


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


        #prefactor = self.Optics.pupilMask*self.Optics.determinant*np.exp(2*np.pi/self.wavelength*1j*self.Optics.pathDifference)
        
       #x_minus = (-1)**np.array(range(self.Optics.ulen))#used to translate ftt to image center
        #ph = np.outer(x_minus, x_minus) #phase required to translate fft to center
        #prefactor *= ph

        E = ifft2(self.prefactor)
        self.Optical_PSF = abs(E)**2
        self.Optical_PSF /= np.sum(self.Optical_PSF*self.dsX*self.dsY) # Normalise to total flux of 1
        #self.Optical_PSF *= np.sum(self.dsX*self.dsY)



        


    def get_E_in_detector(self,filter = interference_filter, detector_thickness=2, zlen=20, nworkers=8):

        ''' Creates self.Ex, self.Ey, self.Ez -- arrays of electric field amplitudes within the detector of thickness tz for self.uX and self.uY. Returns a 3D array of intensity in the postage stamp surrounding the point (SCAx, SCAy) in the SCA and going to a depth of tz. The size of the postage stamp and resolution are determined by ulen.
        Also creates self.Filtered_PSF -- the PSF on the SCA surface after passing through the interference filter, normalised to total flux of 1.
        The interference filter object created earlier is assumed to be the default interference filter.
        '''
        start_time = time.time()
        current_time = time.time()
        #print('Starting get_E_in_detector at time = ',current_time,'\n')
        first_time = time.time()
        z_array = np.linspace(0, detector_thickness, zlen)
        dZ = z_array[1]-z_array[0]
        ulen = self.Optics.ulen
        uX = self.Optics.uX
        uY = self.Optics.uY
        uX, uY = np.meshgrid(uX, uY, indexing='ij')



        E = filter.Transmitted_E(self.wavelength, uX, uY, z_array)
        Ex = E[0]
        Ey = E[1]
        Ez = E[2]

        end_time = time.time()
        print('Time taken to get transmitted E field through filter = ',end_time-current_time,'\n')
        current_time = time.time()
        
        Ex *= self.prefactor[:,:,na]
        Ey *= self.prefactor[:,:, na]
        Ez *= self.prefactor[:,:, na]

        end_time = time.time()
        print('Time taken to multiply by prefactor = ',end_time-current_time,'\n')
        current_time = time.time() 

        Ex_postage_stamp = ifft2(Ex, axes=(0,1),workers=nworkers)
        Ey_postage_stamp = ifft2(Ey, axes=(0,1),workers=nworkers)
        Ez_postage_stamp = ifft2(Ez, axes=(0,1),workers=nworkers)
        #Ex_postage_stamp = np.fft.ifft2(Ex, axes=(0,1))
        #Ey_postage_stamp = np.fft.ifft2(Ey, axes=(0,1))
        #Ez_postage_stamp = np.fft.ifft2(Ez, axes=(0,1))
        end_time = time.time()
        print('Time taken to do ifft = ',end_time-current_time,'\n')
        
        current_time = time.time()
        Intensity = (abs(Ex_postage_stamp)**2) + (abs(Ey_postage_stamp)**2) + (abs(Ez_postage_stamp)**2)

        print('Time taken to compute Intensity by squaring the E field = ', time.time()-current_time,'\n')
        current_time=time.time()
        

        self.Filtered_PSF = Intensity[:,:,0]/np.sum(Intensity[:,:,0]*self.dsX*self.dsY) # Filtered PSF normalise to total flux of 1 (introduced only for testing purposes)
        #self.Filtered_PSF *= np.sum(self.dsX*self.dsY)
        end_time = time.time()
        print('Time taken to calculate Filtered PSF = ',end_time-current_time,'\n')
        current_time=time.time()
        self.Intensity = Intensity
        self.Intensity_integrated = np.trapz(Intensity, x=z_array, axis=2)
        end_time = time.time()
        print('Time taken to integrate over depth = ',time.time()-current_time,'\n')

        print('Total time taken for get_E_in_detector = ',time.time()-start_time,'\n')



        


        return


    def get_detector_image2(self):
        """
        Returns the postage_stamp_size x postage_stamp_size detector image as a 2D array of intensity values.
        """

        #if not hasattr(self, 'Intensity'):
        #    self.get_E_in_detector()

        pix = 10
        # Compute the detector image by summing the contributions from all points in the postage stamp
        #detector_image = np.zeros((, 4088, self.Optics.ulen, self.Optics.ulen), dtype=np.float64)

        XAnalysis, YAnalysis = WFI.fromSCAtoAnalysis(self.Optics.scaNum, self.Optics.scaX, self.Optics.scaY) #Center of the PSF in Analysis coordinates
        
        imageX = XAnalysis + self.sX[:,0]   # Note that self.sX and self.sY are in microns whereas Analysis coordinates and MTF are in mm
        imageY = YAnalysis + self.sY[0,:]
        
        
        Xd = np.floor(XAnalysis//pix)*pix
        Yd = np.floor(YAnalysis//pix)*pix
        xd_array = Xd - (np.floor((self.postage_stamp_size-1)/2)*pix) + pix*np.arange(int(self.postage_stamp_size))
        yd_array = Yd - (np.floor((self.postage_stamp_size-1)/2)*pix) + pix*np.arange(int(self.postage_stamp_size))

        xD, yD = np.meshgrid(xd_array, yd_array, indexing='ij')


        result = MTF_SCA_postage_stamp(imageX, imageY, xD, yD, self.Intensity_integrated, self.npix_boundary)
        self.detector_image2 = result
        
        
    def get_detector_image(self, nworkers=8, chunk_size=1):
        """
        Returns the postage_stamp_size x postage_stamp_size detector image as a 2D array of intensity values.
        """

        #if not hasattr(self, 'Intensity'):
        #    self.get_E_in_detector()

        pix = 10
        # Compute the detector image by summing the contributions from all points in the postage stamp
        #detector_image = np.zeros((, 4088, self.Optics.ulen, self.Optics.ulen), dtype=np.float64)

        XAnalysis, YAnalysis = WFI.fromSCAtoAnalysis(self.Optics.scaNum, self.Optics.scaX, self.Optics.scaY) #Center of the PSF in Analysis coordinates
        
        imageX = XAnalysis + self.sX   # Note that self.sX and self.sY are in microns whereas Analysis coordinates and MTF are in mm
        imageY = YAnalysis + self.sY
        self.imageX = imageX
        self.imageY = imageY

        Xd = np.floor(XAnalysis//pix)*pix
        Yd = np.floor(YAnalysis//pix)*pix
        xd_array = Xd - (np.floor((self.postage_stamp_size-1)/2)*pix) + pix*np.arange(int(self.postage_stamp_size))
        yd_array = Yd - (np.floor((self.postage_stamp_size-1)/2)*pix) + pix*np.arange(int(self.postage_stamp_size))

        xD, yD = np.meshgrid(xd_array, yd_array, indexing='ij')
        mask = (np.maximum(np.abs(xD), np.abs(yD)) <= 20440).astype(np.float64) # Mask to zero out values outside the SCA 
        shape = (int(self.postage_stamp_size), int(self.postage_stamp_size))

        detector_image = np.zeros(shape, dtype=np.float64)

        tasks = [(xd_array[index_xd], yd_array[index_yd], imageX, imageY, self.Intensity_integrated, self.npix_boundary) for index_xd in range(self.postage_stamp_size) for index_yd in range(self.postage_stamp_size)]


        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            results = list(executor.map(parallel_MTF_image, tasks, chunksize=chunk_size))
        
        detector_image = np.array(results).reshape(shape)
         # Mask out values outside the SCA
        detector_image *= mask
        self.detector_image = detector_image
