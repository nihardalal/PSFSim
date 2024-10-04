import numpy as np
import galsim 
from astropy.table import Table
import argparse
import pandas as pd

def makeParser():
    parser = argparse.ArgumentParser(description="Input Files for PSF Simulation")
    parser.add_argument('wcsFileName', help = 'Pathname to Fits File for WCS')
    parser.add_argument('starCat', help = 'Pathname to Star Catalog')
    parser.add_argument('-r', '--random', dest = "randomPos", action = 'store_true', help = 'Use random positions for stars', default = True)
    return parser



if __name__ == '__main__':
    parser=makeParser()
    args=parser.parse_args()

    #Read RA,Dec from star catalog
    try:
        assert('.fits' in args.starCat)
    except:
        raise Exception("Star Catalog should be a .fits file")

    cat = galsim.Catalog(args.starCat)
    
    degrees = galsim.AngleUnit(np.pi/180)
    
    readImage = galsim.fits.read(file_name = args.wcsFileName, hdu = 1)
    mybounds = readImage.bounds
    mywcs = readImage.wcs

    

    outImage = galsim.Image(wcs = mywcs, bounds = mybounds)

    psf = galsim.Moffat(beta=3, fwhm = 2.85)
    source = galsim.Convolve([psf, galsim.DeltaFunction(flux=1.)])

    for i in range(cat.nobjects):
        if not args.randomPos:
            ra = cat.get(i,'RAJ2000')*degrees
            dec = cat.get(i, 'DECJ2000')*degrees

            worldCenter = galsim.CelestialCoord(ra = ra, dec = dec) 
            imageCenter = mywcs.posToImage(worldCenter)
        else:
            x = np.random.random_sample()*mybounds.getXMax()
            y = np.random.random_sample()*mybounds.getYMax()
            imageCenter = galsim.PositionD(x = x, y= y)

        nPhot = 30000 #Need to update as a function of SED for a given star 

        psf.drawImage(outImage, method = 'phot', n_photons = nPhot, center = imageCenter, add_to_image = True)
        print("Image Drawn!")

    outImage.write('outImage.fits')
    print("Image written to outImage.fits")



    



