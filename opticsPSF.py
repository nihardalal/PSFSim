import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from astropy.io import fits
from scipy.linalg import expm
from scipy.interpolate import griddata
from scipy.fft import ifftn

import zernike

#Helper function that computes xout, yout from SCAnum and SCApos here
#Taken from pyimcom config.py https://github.com/kailicao/pyimcom/blob/main/config.py line 121
def fromSCAToPos(SCAnum, SCAx, SCAy):
    xfpa = np.array([-22.14, -22.29, -22.44, -66.42, -66.92, -67.42,-110.70,-111.48,-112.64,
                     22.14,  22.29,  22.44,  66.42,  66.92,  67.42, 110.70, 111.48, 112.64])
    yfpa = np.array([ 12.15, -37.03, -82.06,  20.90, -28.28, -73.06,  42.20,  -6.98, -51.06,
                     12.15, -37.03, -82.06,  20.90, -28.28, -73.06,  42.20,  -6.98, -51.06])
    scIndex = SCAnum-1
    pixsize = 0.01
    nside = 4088
    sca_orient = np.array([-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1]).astype(np.int16)
    if np.amin(SCAnum)<1 or np.amax(SCAnum)>18:
         raise ValueError('Invalid SCA Number')
    return (xfpa[scIndex]+ pixsize*(SCAx-(nside-1)/2.)*sca_orient[scIndex],
            yfpa[scIndex]+ pixsize*(SCAy-(nside-1)/2.)*sca_orient[scIndex], )

class GeometricOptics:
    def __init__(self, SCAnum,SCAx, SCAy, wavelength = 0.48, ulen = 2048):

        self.wavelength = wavelength

        self.scaNum = SCAnum
        self.scaX = SCAx
        self.scaY = SCAy
        #print(self.scaX, self.scaY)

        self.xout, self.yout = fromSCAToPos(self.scaNum, self.scaX, self.scaY)
        self.posOut = np.array([self.xout,self.yout])

        #Set up u,v array for computations of Zernicke Polynomials
        self.ulen = ulen
        self.umin = -1
        self.umax = 1


        self.uX = np.arange(self.umin, self.umax, self.ulen)
        self.uY = np.arange(self.umin, self.umax, self.ulen)

        #Compute Distortion Matrix and dterminant
        self.distortionMatrix = self.computeDistortionMatrix()
        self.determinant = self.computeDeterminant()

        # Obtain pupil Mask and path difference map
        self.pupilMask = self.loadPupilMask()

        self.pathDifference = self.pathDiff()

        #self.integrand = self.pupilMask*self.determinant*expm(2*np.pi/self.wavelength*1j*self.pathDifference)
        #self.x_minus = (-1)**np.array(range(ulen))#used to translate ftt to image center
        #self.ph = np.outer(self.x_minus, self.x_minus) #phase required to translate fft to center
        #self.eArray = np.fft.fft2(self.integrand*self.ph)
        #self.magEArray = abs(self.eArray)

    #Compute distortion matrix here!
    def computeDistortionMatrix(self):

        #Load in polynomial fits to Jacobian
        jacobian_fit_file_name = './data/jacobian_fits.npy'
        self.jacobian_fits = np.load(jacobian_fit_file_name)
        self.wavindex = np.where(self.jacobian_fits['wavelength'] == self.wavelength)
        self.coeff = self.jacobian_fits['coefficients'][self.wavindex]
        self.newpolyorder = self.jacobian_fits['exponents'][self.wavindex]

        return np.sum(self.coeff*np.prod(np.power(self.posOut, self.newpolyorder), axis = 4), axis = 3)[0]
    
    def computeDeterminant(self):

        determinant = self.distortionMatrix[0][1]*self.distortionMatrix[1][0] - self.distortionMatrix[0][0]*self.distortionMatrix[1][1]

        determinant *= 180/np.pi

        return determinant
    
    def loadPupilMask(self):
        dirName = './stpsf-data/WFI/pupils/'
        pupilMaskString = 'SCA{}_full_mask.fits.gz'.format(self.scaNum)
        file = fits.open(dirName+pupilMaskString)
        mask = file[0].data
        return mask
    
    def pathDiff(self):

        self.uArray, self.vArray = np.meshgrid(np.arange(self.umin, self.umax, self.ulen), np.arange(self.umin, self.umax, self.ulen))

        self.urhoPolar =  np.sqrt(self.uArray**2+self.vArray**2)
        self.uthetaPolar = np.arctan2(self.vArray, self.uArray)

        mydata = pd.read_csv('stpsf-data/WFI/wim_zernikes_cycle9.csv', sep=',', header=0)
        #Define mask to desired wavelength and correct X and Y (Need to modify to within bounds):
        mask1 = (mydata['wavelength']==self.wavelength) & (mydata['sca']==self.scaNum)
        #print(np.where(mask1 == True))
        localx = mydata['local_x'][mask1]
        localy = mydata['local_y'][mask1]
        points = np.stack((localx,localy)).T
        #print(points)
        pathDiff = 0
        for i in range(22):
            zIndex = i+1
            zString = ('Z{}'.format(zIndex))
            zernCoeffsToInterpolate = mydata[zString][mask1]
            #print(zernCoeffsToInterpolate)
            zernCoeff = griddata(points, zernCoeffsToInterpolate, (self.scaX,self.scaY), method = 'linear')
            #print(zernCoeff)
            nZern, mZern = zernike.noll_to_zernike(i+1)
            #print(zernike.zernike(nZern, mZern, self.urhoPolar, self.uthetaPolar))
            pathDiff += zernCoeff*zernike.zernike(nZern, mZern, self.urhoPolar, self.uthetaPolar)
        return pathDiff
