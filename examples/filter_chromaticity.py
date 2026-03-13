"""
This is a test script to explore the substrate-induced chromaticity of an
object centroid.


"""

import numpy as np
from psfsim.romantrace import _RomanRayBundle

# first get the bandpasses that we want to study
bandpasses = {
    "R062": (0.480, 0.760),
    "Z087": (0.760, 0.977),
    "Y106": (0.927, 1.192),
    "J129": (1.131, 1.454),
    "H158": (1.380, 1.774),
    "F184": (1.683, 2.000),
    "K213": (1.950, 2.300),
    "W146": (0.927, 2.000),
}
bpkeys = list(bandpasses.keys())

# Now the field positions: currently one at the center of each SCA.
# The format is a numpy array, shape (npos, 2), of xan,yan in degrees.
fpos = np.array(
    [
        [-0.06784, 0.45947],
        [-0.06780, 0.60572],
        [-0.06769, 0.73653],
        [-0.20339, 0.43240],
        [-0.20350, 0.57896],
        [-0.20338, 0.70945],
        [-0.33864, 0.36680],
        [-0.33894, 0.51411],
        [-0.34002, 0.64353],
        [0.06784, 0.45947],
        [0.06780, 0.60572],
        [0.06769, 0.73653],
        [0.20339, 0.43240],
        [0.20350, 0.57896],
        [0.20338, 0.70945],
        [0.33864, 0.36680],
        [0.33894, 0.51411],
        [0.34002, 0.64353],
    ]
)
fpos[:, 1] -= 0.496  # remove field bias

N = 384  # size of grid of rays

# print general information
print("This is a table of the chromatic decenter in each band over the field. We")
print(f"compute the decenter using a ray trace on an {N:d}x{N:d} grid in the pupil")
print("plane, and assign the focal plane position of minimum RMS wavefront error")
print("at each wavelength. The reported position shifts are:\n")
print("    dx = x(blue edge) - x(red edge)   in µm\n")
print("Recall that 1 µm = 0.1 pixel. We use the FPA coordinate system; the sense")
print("of the effect is that the blue image of an object is closer to the center")
print("of the focal plane than the red image.")
print("\n")

# print header
out = "SCA   "
for bp in bpkeys:
    out += f"  --{bp:4s}--     "
print(out)
out = "     "
for _ in bpkeys:
    out += " dx/µm  dy/µm  "
print(out)

# Now let's trace each case.
for sca in range(1, 19):
    # get field position
    xan = fpos[sca - 1, 0]
    yan = fpos[sca - 1, 1]

    # loop over bands
    out = f"{sca:2d} "
    for bp in bpkeys:
        for j in range(2):
            wl = bandpasses[bp][j] / 1000.0  # convert to millimeters
            rays = _RomanRayBundle(xan, yan, N, bp[0], wl=wl)
            u = rays.u[:, :, 0]
            v = rays.u[:, :, 1]
            ds = rays.s - np.sum(rays.open * rays.s) / np.sum(rays.open)  # path length difference in mm

            # Now we want to fit the gradient of path length with respect to (u,v)
            # This is a matrix problem where we want to fit:
            # s ~ c0 + c1*u + c2*v (with least squares)
            sysA = np.zeros((3, 3))
            sysB = np.zeros((3,))
            sysA[0, 0] = np.sum(rays.open)
            sysA[0, 1] = sysA[1, 0] = np.sum(rays.open * u)
            sysA[0, 2] = sysA[2, 0] = np.sum(rays.open * v)
            sysA[1, 1] = np.sum(rays.open * u**2)
            sysA[1, 2] = sysA[2, 1] = np.sum(rays.open * u * v)
            sysA[2, 2] = np.sum(rays.open * v**2)
            sysB[0] = np.sum(rays.open * ds)
            sysB[1] = np.sum(rays.open * ds * u)
            sysB[2] = np.sum(rays.open * ds * v)
            coefs = np.linalg.solve(sysA, sysB) * 1000.0  # convert to microns

            # get position at blue - red end of bandpass
            # note the - sign: ∂s/∂u < 0 if the minimum WFE position is farther to the *right* (dx>0).
            if j == 0:
                dx = -coefs[1]
                dy = -coefs[2]
            else:
                dx += coefs[1]
                dy += coefs[2]

        out += f"  {dx:6.3f} {dy:6.3f}"

    print(out)
