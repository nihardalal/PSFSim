import numpy as np


def fromSCAToFPA(SCAnum, SCAx, SCAy):
    """
    Coordinate transformation converting SCA coordinates (in pixels) to FPA coordinates (in mm)
    """
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

def fromSCAtoAnalysis(SCAnum, SCAx, SCAy):
    """
    Coordinate transformation converting SCA coordinates (in pixels) to Analysis coordinates (in microns). The Analysis coordinates system is defined to be the FPA coordinate system with origin shifted to the center of the SCA
    """
    scIndex = SCAnum-1
    pixsize = 10 # microns
    nside = 4088
    sca_orient = np.array([-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1]).astype(np.int16)
    if np.amin(SCAnum)<1 or np.amax(SCAnum)>18:
         raise ValueError('Invalid SCA Number')
    return (pixsize*(SCAx-(nside-1)/2.)*sca_orient[scIndex], pixsize*(SCAy-(nside-1)/2.)*sca_orient[scIndex])