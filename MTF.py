import numpy as np


def MTF(x, y):
    """
    Charge diffusion modulation transfer function as a function of SCA coordinates SCAx and SCAy which should be in microns.
    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA, 
    where the parameters are derived from the charge diffusion model
    described in Emily's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632
    """
    pix=10 # pixel size in microns
    sigma_s = 0.3279*pix # sigma of the charge diffusion in microns
    w1 = 0.17519
    w2 = 0.53146
    w3 = 0.29335
    c1 = 0.4522
    c2 = 0.8050
    c3 = 1.4329

    sigma1 = c1*sigma_s
    sigma2 = c2*sigma_s
    sigma3 = c3*sigma_s

    MTF1 = w1 * np.exp(-((x**2 + y**2) / (2 * sigma1**2))) * (1/(2*np.pi*sigma1**2))
    MTF2 = w2 * np.exp(-((x**2 + y**2) / (2 * sigma2**2))) * (1/(2*np.pi*sigma2**2))
    MTF3 = w3 * np.exp(-((x**2 + y**2) / (2 * sigma3**2))) * (1/(2*np.pi*sigma3**2))
    MTF_total = MTF1 + MTF2 + MTF3
    return MTF_total
