import numpy as np
import galsim 
from astropy.table import Table
import argparse
import pandas as pd
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
import astropy.io as aio
from astropy import constants as const

class imageCube():

    def __init__(self, image2D, tExp = 120):
        self.image2D = image2D
        self.imageCube = self.makeImageCube
        self.tFrame = 3.08
        self.tExp = tExp
        self.nFrames = self.tExp//self.tFrame

    def makeImageCube(self):
        rng = np.random.default_rng()
        pArray = [1/(self.nFrames-1)]*(self.nFrames-1)
        rvs = rng.multinomial(self.image2D, pArray, size = (self.nFrames-1,) +self.image2D.shape)
        return np.cumsum(rvs)
