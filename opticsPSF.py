import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from astropy.io import fits
from scipy.linalg import expm

import zernike

#Helper function that computes SCAnum and SCApos from xout yout here
def fromPosToSCA(x,y):
    return 
#Helper function that computes xout, yout from SCAnum and SCApos here
def fromSCAToPos(SCAnum, SCApos):
    return 

class GeometricOptics:
    def __init__(self, xout,yout, wavelength = 0.48, ulen = 2048):

        self.xout = xout
        self.yout = yout
        self.posOut = np.array([self.xout,self.yout])
        self.wavelength = wavelength


        self.scaNum, self.scaPos = fromPosToSCA(self.xout, self.yout)

        #Set up u,v array for computations of Zernicke Polynomials
        self.ulen = ulen
        self.umin = -1
        self.umax = 1
        self.uArray, self.vArray = np.meshgrid(np.arange(self.umin, self.umax, self.ulen), np.arange(self.umin, self.umax, self.ulen))

        self.urhoPolar =  np.sqrt(self.uArray**2+self.vArray**2)
        self.uthetaPolar = np.arctan2(self.vArray, self.uArray)

        #Load in polynomial fits to Jacobian
        jacobian_fit_file_name = './data/jacobian_fits.npy'
        self.jacobian_fits = np.load(jacobian_fit_file_name)
        self.wavindex = np.where(self.jacobian_fits['wavelength'] == self.wavelength)
        self.coeff = self.jacobian_fits['coefficients'][self.wavindex]
        self.newpolyorder = self.jacobian_fits['exponents'][self.wavindex]

        #Compute Distortion Matrix and dterminants
        self.distortionMatrix = self.computeDistortionMatrix()
        self.determinant = self.computeDeterminant()
        self.determinant*=180/np.pi

        self.pupilMask = self.loadPupilMask()

        self.pathDifference = self.pathDiff()

        self.integrand = self.pupilMask*self.determinant*expm(2*np.pi/self.wavelength*1j*self.pathDifference)


    #Compute distortion matrix here!
    def computeDistortionMatrix(self):
        return np.sum(self.coeff*np.prod(np.power(np.array(self.posOut), self.newpolyorder), axis = 3), axis = 2)
    
    def computeDeterminant(self):
        return self.distortionMatrix[0][1]*self.distortionMatrix[1][0] - self.distortionMatrix[0][0]*self.distortionMatrix[1][1]
    
    def loadPupilMask(self):
        dirName = './stpsf-data/WFI/pupils'
        pupilMaskString = 'SCA{}_full_mask.fits.gz'.format(self.scaNum)
        file = fits.open(dirName+pupilMaskString)
        mask = file[0].data
        return mask
    
    def pathDiff(self):
        mydata = pd.read_csv('stpsf-data/WFI/wim_zernikes_cycle9.csv', sep=',', header=0)
        #Define mask to desired wavelength and correct X and Y (Need to modify to within bounds):
        mask1 = np.where(mydata['wavelength']==self.wavelength and mydata['globalX'] == self.xout and mydata['globalY'] == self.yout)
        pathDiff = 0
        for i in range(22):
            zIndex = i+1
            zString = ('Z{}'.format(zIndex))
            zernCoeff = mydata[zString][mask1]
            nZern, mZern = zernike.noll_to_zernike(i)
            pathDiff += zernCoeff*zernike.zernike(nZern, mZern, self.urhoPolar, self.uthetaPolar)
        return pathDiff

        







