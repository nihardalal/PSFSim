import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#Helper function that computes SCAnum and SCApos from xout yout here
def fromPosToSCA(x,y):
    return 
#Helper function that computes xout, yout from SCAnum and SCApos here
def fromSCAToPos(SCAnum, SCApos):
    return 

class GeometricOptics:
    def __init__(self, xout,yout, wavelength = 0.48):

        self.xout = xout
        self.yout = yout
        self.posOut = np.array([self.xout,self.yout])
        self.wavelength = wavelength

        self.scaNum, self.scaPos = fromPosToSCA(self.xout, self.yout)

        #Jacobian polynomial fit tolerance
        self.tolerance = 1e-4

        #Read in data file from stpsf-data
        self.dataFile = 'stpsf-data/WFI/wim_zernikes_cycle9.csv'
        self.dataframe = pd.read_csv(self.dataFile, sep=',', header=0)
        
        self.wavmask = np.where(self.dataframe['wavelength']==self.wavelength)
        self.localAngleXArray = np.array(self.dataframe['axis_local_angle_x'])[self.wavmask]
        self.localAngleYArray = np.array(self.dataframe['axis_local_angle_y'])[self.wavmask]
        self.globalXArray = np.array(self.dataframe['global_x'])[self.wavmask]
        self.globalYArray = np.array(self.dataframe['global_y'])[self.wavmask]

        #Fit polynomial function to data - ideally only want to do this once, so will probably move this outside and save the polynomial fit as a numpy file. 
        angleInput = np.dstack(self.localAngleXArray, self.localAngleYArray)
        poly = PolynomialFeatures(degree=4)
        angleInput_ = poly.fit_transform(angleInput[0])
        modelX = linear_model.LinearRegression()
        modelY = linear_model.LinearRegression()
        modelX.fit(angleInput_, self.globalXArray)
        modelY.fit(angleInput_, self.globalYArray)

        rx = modelX.score(angleInput_, self.globalXArray)
        ry = modelY.score(angleInput_, self.globalYArray)
        print("rx,ry =", rx, ry)
        if 1-rx>self.tolerance or 1-ry>self.tolerance:
            raise Exception("Not able to find a good model fit, please try again with higher polynomial order")
        
        coefX = modelX.coef_
        coefY = modelY.coef_
        xpowers = poly.powers_[:,0]
        ypowers = poly.powers_[:,1]
        newpowersx = np.clip(xpowers-1,0,None)
        newpowersy = np.clip(ypowers-1,0,None)

        self.newpolyorder = np.empty((2,2, 15, 2), dtype = object)
        self.newpolyorder[0][0] = np.stack((newpowersx,ypowers), axis = 1)
        self.newpolyorder[0][1] = np.stack((xpowers,newpowersy), axis = 1)
        self.newpolyorder[1][0] = np.stack((newpowersx,ypowers), axis = 1)
        self.newpolyorder[1][1] = np.stack((xpowers,newpowersy), axis = 1)

        self.jacobian = np.empty((2,2,15), dtype = object)
        self.jacobian[0][0] = xpowers*coefX
        self.jacobian[0][1] = ypowers*coefX
        self.jacobian[1][0] = xpowers*coefY
        self.jacobian[1][1] = ypowers*coefY


    #Compute distortion matrix here!
    def distortionMatrix(self):
        return np.sum(self.jacobian*np.prod(np.power(np.array(self.posOut), self.newpolyorder), axis = 3), axis = 2)
    
    def pupilMask(self):
        return 
    
    def zernicke(self):
        return 

        







