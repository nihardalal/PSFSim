import numpy as np
import galsim 
from astropy.table import Table
import argparse
import pandas as pd

def makeParser():
    parser = argparse.ArgumentParser(description="Input Files for PSF Simulation")
    parser.add_argument('wcsFileName', help = 'Pathname to Fits File for WCS')
    parser.add_argument('starCat', help = 'Pathname to Star Catalog')

    return parser



if __name__ == '__main__':
    parser=makeParser()
    args=parser.parse_args()

    #Read RA,Dec from star catalog
    try:
        assert('.txt' in args.starCat)
    except:
        raise Exception("Star Catalog should be a .txt file")
    
    #starCat = np.loadtxt(args['starCat'])

    table = Table.read(args.starCat, format = 'ascii')
    
    degrees = galsim.AngleUnit(np.pi/180)
    ra = table['RAJ2000']*degrees
    dec = table['DECJ2000']*degrees

    worldPos = [galsim.CelestialCoord(ra = ra[i], dec = dec[i]) for i in range(len(ra))]

    mywcs = galsim.FitsWCS(args.wcsFileName)

    pixelPos = [mywcs.posToImage(worldPos[i]) for i in range(len(worldPos))]

    print(pixelPos[0])

