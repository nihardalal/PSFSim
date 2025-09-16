import numpy as np


def MTF(x1, y1, x2, y2):
    """
    Charge diffusion modulation transfer function as a function of Analysis coordinates in mm.
    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA, 
    where the parameters are derived from the charge diffusion model
    described in Emily's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632
    Note that in terms of the arguments x1, y1, x2, y2 the MTF computes diffusion from (x2, y2) to (x1, y1). 
    """

    deltax = x1 - x2
    deltay = y1 - y2
    pix = 0.01  # pixel size in mm
    sigma_s = 0.3279*pix # sigma of the charge diffusion in pixel units
    w1 = 0.17519
    w2 = 0.53146
    w3 = 0.29335
    c1 = 0.4522
    c2 = 0.8050
    c3 = 1.4329

    sigma1 = c1*sigma_s
    sigma2 = c2*sigma_s
    sigma3 = c3*sigma_s

    MTF1 = w1 * np.exp(-((deltax**2 + deltay**2) / (2 * sigma1**2))) * (1/(2*np.pi*sigma1**2))
    MTF2 = w2 * np.exp(-((deltax**2 + deltay**2) / (2 * sigma2**2))) * (1/(2*np.pi*sigma2**2))
    MTF3 = w3 * np.exp(-((deltax**2 + deltay**2) / (2 * sigma3**2))) * (1/(2*np.pi*sigma3**2))
    MTF_total = MTF1 + MTF2 + MTF3
    return MTF_total

def MTF_SCA(x, y, npix_boundary=1):
    """
    Modulation Transfer Function (MTF) for diffusion from point in SCA with Analysis coordinates (x, y) to pixel coordinates given by integer pairs (i, j). 0 <= i,j < 4088. Reflection boundary conditions are applied at the edges of the SCA.
    The MTF is computed for all pixels in the SCA, and the result is returned as a 2D array of shape (4088, 4088).
    """
    pix = 0.01  # pixel size in mm
    nside = 4088 # number of active pixels per side in the SCA (Should this be 4088 or 4096?)
    side_length = nside * pix  # Length of the SCA in mm
    xp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)  # x coordinates of pixel centers in mm
    yp_array = (-side_length/2) + (np.array(range(nside)) * pix + pix/2)  # y coordinates of pixel centers in mm
    Xp, Yp = np.meshgrid(xp_array, yp_array, indexing='ij')  # Create a meshgrid of pixel centers
    MTF_SCA_array = np.zeros((nside, nside))

    if not max(abs(x), abs(y)) < side_length/2:
        print('point is outside the SCA')
        return MTF_SCA_array

    else:


        MTF_SCA_array = MTF(Xp, Yp, x, y) ## Baseline MTF without any boundary layer corrections

        # Check if the point is in the left/right boundary layer and apply reflection boundary conditions if necessary
        if x < -side_length/2 + npix_boundary * pix :
            # point is in the left boundary layer
            x_reflected = x - 2 * (x + side_length/2)
            MTF_SCA_array += MTF(Xp, Yp, x_reflected, y)

        elif x > side_length/2 - npix_boundary * pix:
            # point is in the right boundary layer
            x_reflected = x + 2 * (side_length/2 - x)
            MTF_SCA_array += MTF(Xp, Yp, x_reflected, y)
        
        # Check if the point is in the top/bottom boundary layer and apply reflection boundary conditions if necessary
        if y < -side_length/2 + npix_boundary * pix:
            # point is in the bottom boundary layer
            y_reflected = y - 2 * (y + side_length/2)
            MTF_SCA_array += MTF(Xp, Yp, x, y_reflected)

        elif y > side_length/2 - npix_boundary * pix:
            # point is in the top boundary layer
            y_reflected = y + 2 * (side_length/2 - y)
            MTF_SCA_array += MTF(Xp, Yp, x, y_reflected)



        return MTF_SCA_array   
    
    

