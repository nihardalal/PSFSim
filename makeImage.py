import numpy as np
import galsim 
from astropy.table import Table
import argparse
import pandas as pd
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
import astropy.io as aio
from astropy import constants as const

fNuRef = 3.631e-23*(u.W/u.m**2)/u.Hz#W/m^2/Hz

def makeParser():
    parser = argparse.ArgumentParser(description="Input Files for PSF Simulation")
    parser.add_argument('wcsFileName', help = 'Pathname to Fits File for WCS')
    parser.add_argument('starCat', help = 'Pathname to Star Catalog')
    parser.add_argument('SCA', help = "SCA Number")
    parser.add_argument('-r', '--random', dest = "randomPos", action = 'store_true', help = 'Use random positions for stars', default = False)
    parser.add_argument('-b', '--blackbody', dest ="blackBody", action = 'store_true', help = 'Each star is assumed to be 5000 K blackbody for testing purposes', default = False)
    return parser

def sedBB(w, T):
    return ((8*np.pi*const.h*const.c**2/w**5)*1/(np.exp(const.h*const.c/(w*const.k_B*T))-1)).decompose()

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

    scaNum = int(args.SCA)
    if scaNum <10:
        effAreaTable = aio.ascii.read('Roman_effarea_tables_20240327/Roman_effarea_v8_SCA0{}_20240301.ecsv'.format(scaNum))
    else:
        effAreaTable = aio.ascii.read('Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{}_20240301.ecsv'.format(scaNum))

    mirrorDiameter = 2.37*u.m

    geomArea = np.pi*mirrorDiameter**2/4
    transmissionCurve = effAreaTable['F184']*u.m**2/geomArea

    tExp = 120*u.s

    outImage = galsim.Image(wcs = mywcs, bounds = mybounds)

    psf = galsim.Moffat(beta=3, fwhm = 2.85)
    source = galsim.Convolve([psf, galsim.DeltaFunction(flux=1.)])

    for i in range(cat.nobjects):
        if not args.randomPos:
            ra = cat.get(i,'RAJ2000')*degrees
            dec = cat.get(i, 'DECJ2000')*degrees
            mag = cat.get(i, 'H')

            worldCenter = galsim.CelestialCoord(ra = ra, dec = dec) 
            imageCenter = mywcs.posToImage(worldCenter)
        else:
            x = np.random.random_sample()*mybounds.getXMax()
            y = np.random.random_sample()*mybounds.getYMax()
            imageCenter = galsim.PositionD(x = x, y= y)
        
        if args.blackBody:
            #bb = BlackBody(temperature = 5000*u.K)
            wav = np.arange(0.400,2.600, 0.001) * u.um
            fluxUnnorm = sedBB(wav, 5000*u.K)
            fLambdaRef = fNuRef * const.c / wav**2
            norm = 10**(-0.4*mag) * np.trapz(fLambdaRef*transmissionCurve*wav, x = wav)/np.trapz(fluxUnnorm*transmissionCurve*wav, x = wav)
            flux = norm*fluxUnnorm
            nPhotQ = np.trapz(flux*effAreaTable['F184']*u.m**2*wav*tExp/(const.h * const.c), x= wav)
            nPhotQ = nPhotQ.decompose()
            #nPhotQ = nPhotQ.to(1/(u.s*u.cm**2))
            nPhot = nPhotQ.value
            print(nPhot)
        else:
            nPhot = 300000 #Need to update as a function of SED for a given star 

        geomAreaCM = geomArea.to(u.cm**2).value

        psf.drawImage(outImage, method = 'phot', n_photons = nPhot, center = imageCenter, add_to_image = True)
        print("Image Drawn!")

    outImage.write('outImage.fits')
    print("Image written to outImage.fits")



    



