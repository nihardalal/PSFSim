import numpy as np
import scipy
from astropy.io import fits

from .mirror_properties import reflect_RB_model
from .offsets import fbias_offset, fpa_offset, sm_offset
from .wfi_data import scapos

fratio_scale = 7.935  # scale FPA displacement modes by 8*f^2
# based on the paraxial EFL = 18751.5 mm and the PM baffle inner radius of 1181.56 mm


def _shadow(arr):
    """This is for testing only. Patch this function to block some of the light paths."""
    pass


def _lanczos_weight(dx, dy, a=3):
    """
    Lanczos function.

    Parameters
    ----------
    dx : float or np.ndarray
        The input x value(s).
    dy: float or np.ndarray,
        The input y value(s). Must be the same shape as `dx`.
    a : int, optional
        The Lanczos parameter (default is 3).

    Returns
    -------
    float or np.ndarray
        The Lanczos function evaluated on the `dx` by `dy` grid.

    """

    r_x = np.where(np.abs(dx) < a, np.sinc(dx) * np.sinc(dx / a), 0.0)
    r_y = np.where(np.abs(dy) < a, np.sinc(dy) * np.sinc(dy / a), 0.0)
    return r_x * r_y


def _apply_lanczos_reweighting(
    RB_open_lowres,
    RB_open_hires,
    bdycells,
    ovsamp,
    a_lanczos,
):
    """
    Apply Lanczos reweighting to boundary cells using high-resolution data.

    Parameters
    ----------
    RB_open_lowres : ndarray
        Low-resolution aperture array
    RB_open_hires : ndarray
        High-resolution aperture array (flattened to num_bdy, ovsamp, ovsamp)
    bdycells : tuple
        (x_indices, y_indices) of boundary cells
    ovsamp : int
        Oversampling factor
    a_lanczos : int
        Lanczos kernel size parameter

    Returns
    -------
    new_values : ndarray
        Reweighted aperture values at boundary cells
    """

    sub_offsets = np.linspace(-0.5 + 0.5 / ovsamp, 0.5 - 0.5 / ovsamp, ovsamp)
    sx, sy = np.meshgrid(sub_offsets, sub_offsets, indexing="ij")
    m_lanczos = 2 * a_lanczos + 1
    # Trying updated Lanczos scheme to weight more pixels without simulating more at hires
    dx_arr = np.arange(-a_lanczos, a_lanczos + 1)
    dy_arr = np.arange(-a_lanczos, a_lanczos + 1)
    W_sub = np.zeros((m_lanczos, m_lanczos, ovsamp, ovsamp))

    for i, dx in enumerate(dx_arr):
        for j, dy in enumerate(dy_arr):
            w = _lanczos_weight((dx + sx).ravel(), (dy + sy).ravel(), a=a_lanczos)
            W_sub[i, j] = w.reshape(ovsamp, ovsamp)

    # Normalize weights so that full filter sums to 1
    W_sub /= np.sum(W_sub)

    # Calculate low res weights for non sub-sampled pixels
    W_low = np.sum(W_sub, axis=(2, 3))

    pad_w = a_lanczos
    RB_open_padded = np.pad(RB_open_lowres, pad_w, mode="edge")

    # Map from x,y in spatial coordinates to index in bdycells
    hires_index_map = np.full(RB_open_lowres.shape, -1, dtype=np.int32)
    hires_index_map[bdycells[0], bdycells[1]] = np.arange(len(bdycells[0]))
    hires_map_padded = np.pad(hires_index_map, pad_w, constant_values=-1)

    num_bdy = len(bdycells[0])
    hires_open_reshaped = RB_open_hires.astype(np.float64).reshape(num_bdy, ovsamp, ovsamp)

    new_values = np.zeros(num_bdy, dtype=np.float64)
    bx = bdycells[0] + pad_w
    by = bdycells[1] + pad_w

    for i, dx in enumerate(dx_arr):
        for j, dy in enumerate(dy_arr):
            nx = bx + dx
            ny = by + dy

            k_prime = hires_map_padded[nx, ny]
            is_hires = k_prime >= 0

            contrib = np.zeros(num_bdy, dtype=np.float64)
            contrib[~is_hires] = RB_open_padded[nx[~is_hires], ny[~is_hires]] * W_low[i, j]

            if np.any(is_hires):
                H = hires_open_reshaped[k_prime[is_hires]]
                contrib[is_hires] = np.sum(H * W_sub[i, j], axis=(1, 2))

            new_values += contrib

    return new_values


### begin material data ###

NORM_TOL = 1e-8


def n_Infrasil301(wl, T=180.0):
    """
    Function to compute the index of refraction of Infrasil 301.

    Parameters
    ----------
    wl : float
        Vacuum wavelength in millimeters.
    T : float, optional
        Temperature in Kelvin.

    Returns
    -------
    float
        The (real) index of refraction.

    Notes
    -----
    Original reference is:
    Sellmeier coefficients from Leviton, Frey, & Madison (2008)
    arXiv:0805.0096

    Valid in the Roman wavelength range.

    """

    data = np.array(
        [
            [0.105962, 0.995429, 0.865120, 4.500743e-03, 9.383735e-02, 9.757183],
            [9.359142e-06, -7.973196e-06, 3.731950e-04, -2.825065e-04, -1.374171e-06, 1.864621e-03],
            [4.941067e-08, 1.006343e-09, -2.010347e-06, 3.136868e-06, 1.316037e-08, -1.058414e-05],
            [4.890163e-11, -8.694712e-11, 2.708606e-09, -1.121499e-08, 1.252909e-11, 1.730321e-08],
            [1.492126e-13, -1.220612e-13, 1.679976e-12, 1.236514e-11, -4.641280e-14, 1.719396e-12],
        ]
    )
    D = data.T @ np.array([1, T, T**2, T**3, T**4])
    L = wl * 1e3  # convert to microns
    return np.sqrt(1 + np.sum(D[:3] * L**2 / (L**2 - D[-3:] ** 2)))


### end material data ###


def build_transform_matrix(xde=0.0, yde=0.0, zde=0.0, ade=0.0, bde=0.0, cde=0.0, unit="degree"):
    """
    Makes a transformation matrix from a set of translations and rotations.

    The matrix is 4x4 and the convention is::

        u(global coords) = R u(obj coords)
        where u is a vector of [1,x,y,z]

    Parameters
    ----------
    xde, yde, zde : float, optional
        Translations on each axis. The convention is that the origin in object coordinates
        is at (`xde`, `yde`, `zde`) in global coordinates.
    ade, bde, cde : float, optional
        Euler angles, in the rotX - rotY - rotZ convention (so you can envision rotating the global to
        the object system via a right-handed rotation around X; then a right-handed rotation around Y;
        and finally a left-handed rotation around Z). The convention is that:

        - The *object* z-axis direction is in the *global* direction
          (-sin bde, sin ade * cos bde, cos ade * cos bde)

        - The *global* x-axis direction is in the *object* direction
          (cos bde * cos cde, cos bde * sin cde, -sin bde).
    unit : str, optional
        Angular unit: "degree" (default) or "radian".

    Returns
    -------
    np.ndarray
        The 4x4 transformation matrix.

    """

    # conversion
    if unit.lower()[:3] == "deg":
        ade *= np.pi / 180.0
        bde *= np.pi / 180.0
        cde *= np.pi / 180.0

    R = np.zeros((4, 4))
    R[0, 0] = 1.0
    R[1, 0] = xde
    R[2, 0] = yde
    R[3, 0] = zde

    # the [1:,1:] sub-block is the rotation matrix
    R[1:, 1:] = (
        np.array([[1, 0, 0], [0, np.cos(ade), np.sin(ade)], [0, -np.sin(ade), np.cos(ade)]])
        @ np.array([[np.cos(bde), 0, -np.sin(bde)], [0, 1, 0], [np.sin(bde), 0, np.cos(bde)]])
        @ np.array([[np.cos(cde), -np.sin(cde), 0], [np.sin(cde), np.cos(cde), 0], [0, 0, 1]])
    )
    return R


# start with RB.xyFPA vs RB.u
def xyFPA_from_u(RB, thres):
    """
    Performs a linear regression of xyFPA vs u for a given RayBundle object and open threshold;
    returns a least squares coefficient array as a dictionary.

    Arguments
    ---------
    RB: RayBundle object
        ray bundle of choice
    thres: int
        open ray threshold (fraction of rays in bundle that hit detector)... between 0 and 1

    Returns
    --------
    coeff: dict
        coefficient array as a dictionary, where:
            coeff["Slope"] is a (2, 2) matrix
            coeff["Intercept"] is a (2,) vector

    """
    open_rays = np.where(RB.open > thres)
    u = RB.u[:, :, :][open_rays]
    xyFPA = RB.xyFPA[:, :, :][open_rays]
    u_stacked = np.hstack([u, np.ones([u.shape[0], 1], u.dtype)])
    M = np.linalg.lstsq(u_stacked, xyFPA, rcond=None)[0]
    A = M[0:2, :]
    b = M[2, :]
    coeff = {}
    coeff["Slope"] = A
    coeff["Intercept"] = b
    return coeff


# next RB.u vs RB.xyi
def u_from_xyi(RB, thres):
    """
    Performs a linear regression of u vs xyi for a given RayBundle object and open threshold;
    returns a least squares coefficient array as a dictionary.

    Arguments
    ---------
    RB: RayBundle object
        ray bundle of choice
    thres: int
        open ray threshold (fraction of rays in bundle that hit detector)... between 0 and 1

    Returns
    --------
    coeff: dict
        coefficient array as a dictionary, where:
            coeff["Slope"] is a (2, 2) matrix
            coeff["Intercept"] is a (2,) vector
    """
    open_rays = np.where(RB.open > thres)
    xyi = RB.xyi[:, :, :][open_rays]
    u = RB.u[:, :, :][open_rays]
    xyi_stacked = np.hstack([xyi, np.ones([u.shape[0], 1], u.dtype)])
    M = np.linalg.lstsq(xyi_stacked, u, rcond=None)[0]
    A = M[0:2, :]
    b = M[2, :]
    coeff = {}
    coeff["Slope"] = A
    coeff["Intercept"] = b
    return coeff


class RayBundle:
    """
    Class defining a ray bundle, constructed from a field position.

    Parameters
    ----------
    xan, yan : float
        Field angles (in degrees).
    N : int
        The bundle size on each axis of the entrance pupil (so ``N**2`` total points).
    hasE : bool, optional
        Also propagate the electric field?
    width : float, optional
        The size of the grid for the entrance pupil (in mm; default of 2500 mm is
        good for Roman).
    startpos : float, optional
        Starting Z-position in mm (default is before the SM obstruction in Roman).
    wl : float, optional
        The vacuum wavelength in mm.
    wlref : float, optional
        The vacuum wavelength in mm used as a reference (this is so that when we make
        chromatic PSFs, the "center" of the PSF always ends up in the same place, and
        e.g. DCR effects show up in wavelength-dependent Zernike Z2 & Z3 modes).
    jacobian : np.ndarray, optional
        If used, this will give a 2x2 distortion matrix, used so that the output exit pupil is
        on a square grid. Default is a square grid on the entrance pupil.
    hires : list of np.ndarray of int, optional
        If given, then instead of the full grid, trace only the cells given;
        ``hires[0]`` is a 1D array of y-values and ``hires[1]`` is a 1D array of x-values.
    ovsamp : int, optional
        Oversamples cells in the entrance pupil by this factor; only used if `hires` is given.
    idealgeom : bool, optional
        Forces the design model rather than with the best-fit offsets.
    save_bundle_history : bool, optional
        Adds location_history and momentum_history attributes to the ray bundle so it can track
        its locations and momenta along its path.

    Attributes
    ----------
    N : int
        Size of ray bundle.
    N1, N2: int
        Trimmed size (needed for hires mode; otherwise equal to `N`).
    x : np.ndarray of float
        3D array of position of rays, shape (`N1`, `N2`, 4); unit = mm;
        the last axis is a position so has ``x[:,:,0] == 1``.
    p : np.ndarray of float
        3D array of direction of rays, shape (`N1`, `N2`, 4);
        the last axis is a direction so has ``p[:,:,0] == 0``.
    s : np.ndarray of float
        2D array of path length, shape (`N1`, `N2`); unit = mm.
    n_loc : float
        Local index of refraction.
    xan, yan: float
        Field angles (in degrees).
    costhetaent: float
        Projection factor for entrance aperture.
    open : np.ndarray of bool
        Has this ray propagated?
    wl, wlref : float
        Wavelength and reference wavelength (the latter for geometric trace).
        Units are mm.
    E : np.ndarray of complex or None.
        The electric field (optional, 4D: 2D for array, 1D for input pol, 1D for output pol).
        Shape is (`N1`, `N2`, 2, 2).
        None if not used.
    location_history : list
        A list of ray positions (from self.x) along the ray bundle's total path.
    momentum_history : list
        A list of ray momenta (from self.p) along the ray bundle's total path.

    Methods
    -------
    intersect_surface
        Gets intersection of a ray bundle and a conic section surface.
    mask
        Masks incoming rays at a given surface.
    intersect_surface_and_reflect
        Propagates rays to a surface and performs a reflection.
    intersect_surface_and_refract
        Propagates rays to a surface and performs a refraction.
    pad
        Pads the ray bundle to a target size, centering the current data.
    fit
        Does a 2D linear fit to the bundle from a given field point.

    """

    # forward and reverse transformations
    @classmethod
    def MV(cls, a, b):
        """
        Matrix-vector multiplication.

        Parameters
        ----------
        a : np.ndarray
            Matrix, shape (4, 4).
        b : np.ndarray
            Array of vectors, shape (..., 4).

        Returns
        -------
        np.ndarray
            Product `a` times `b`, acting on last axis, same shape as `b`.

        See Also
        --------
        MiV
            This is the inverse function.

        """

        return np.einsum("ij,...j->...i", a, b)

    @classmethod
    def MiV(cls, a, b):
        """
        Inverse matrix-vector multiplication.

        Parameters
        ----------
        a : np.ndarray
            Matrix, shape (4, 4).
        b : np.ndarray
            Array of vectors, shape (..., 4).

        Returns
        -------
        np.ndarray
            Product inv(`a`) times `b`, acting on last axis, same shape as `b`.

        See Also
        --------
        MV
            The forward function (``MiV`` is the inverse).

        """

        a_ = np.linalg.inv(a)
        a_[0, 0] = 1.0
        a_[0, 1:] = 0.0
        return np.einsum("ij,...j->...i", a_, b)

    def __init__(
        self,
        xan,
        yan,
        N,
        hasE=False,
        width=2500.0,
        startpos=3500.0,
        wl=1.29e-3,
        wlref=1.29e-3,
        jacobian=None,
        hires=None,
        ovsamp=6,
        idealgeom=True,
        save_bundle_history=False,
    ):
        if jacobian is None:
            jacobian = np.array([[1, 0], [0, 1]])

        self.N = N
        self.xan = xan
        self.yan = yan
        self.n_loc = 1.0

        # make a grid for the entrance pupil
        s = np.linspace(-width / 2.0 * (1 - 1.0 / N), width / 2.0 * (1 - 1.0 / N), N)
        xi, yi = np.meshgrid(s, s)
        self.N1 = self.N2 = N

        # hi-resolution mode, if requested
        if hires is not None:
            sf = np.linspace(-0.5 * (1 - 1.0 / ovsamp), 0.5 * (1 - 1.0 / ovsamp), ovsamp) * width / N
            xf, yf = np.meshgrid(sf, sf)
            xf = xf.ravel()
            yf = yf.ravel()
            self.N1 = np.size(hires[1])
            self.N2 = ovsamp**2
            xi = np.zeros((self.N1, self.N2))
            yi = np.zeros((self.N1, self.N2))
            xi[:, :] = s[hires[1]][:, None] + xf[None, :]
            yi[:, :] = s[hires[0]][:, None] + yf[None, :]

        self.x = np.zeros((self.N1, self.N2, 4))
        self.p = np.zeros((self.N1, self.N2, 4))
        self.open = np.ones((self.N1, self.N2), dtype=bool)

        # build initial position vector
        self.x[:, :, 0] = 1.0
        self.x[:, :, 1] = jacobian[0][0] * xi + jacobian[0][1] * yi
        self.x[:, :, 2] = jacobian[1][0] * xi + jacobian[1][1] * yi
        self.x[:, :, 3] = startpos

        xi = self.x[:, :, 1]
        yi = self.x[:, :, 2]

        self.xyi = self.x[:, :, 1:3]

        # get directions
        r = np.sqrt(xan**2 + yan**2) * np.pi / 180.0
        self.costhetaent = np.cos(r)
        phi = np.arctan2(yan, -xan)
        self.p[:, :, 1] = -np.sin(r) * np.cos(phi)
        self.p[:, :, 2] = -np.sin(r) * np.sin(phi)
        self.p[:, :, 3] = -np.cos(r)

        self.wl = wl
        self.wlref = wlref

        # initial path length
        self.s = xi * self.p[:, :, 1] + yi * self.p[:, :, 2]

        # remove field bias, rotate to Payload Coordinate System
        field_bias = build_transform_matrix(ade=-0.496, cde=150.0, unit="degree")
        if not idealgeom:
            field_bias = (
                build_transform_matrix(
                    ade=fbias_offset["VERTICAL"], bde=fbias_offset["HORIZONTAL"], unit="degree"
                )
                @ field_bias
            )
        # the following transformations take you from WFI field coordinates to PCS coordinates
        self.x = RayBundle.MiV(field_bias, self.x)
        self.p = RayBundle.MiV(field_bias, self.p)

        self.save_bundle_history = save_bundle_history
        if save_bundle_history:
            self.location_history = [self.x[:,:,1:4].copy()]
            self.momentum_history = [self.p[:,:,1:4].copy()]

        # if requested, build the E-field
        if hasE:
            self.E = np.zeros((N, N, 2, 4), dtype=np.complex128)
            # ingoing E in x-pol
            self.E[:, :, 0, 1] = np.sin(phi) ** 2 + np.cos(r) * np.cos(phi) ** 2
            self.E[:, :, 0, 2] = (np.cos(r) - 1) * np.sin(phi) * np.cos(phi)
            self.E[:, :, 0, 3] = -np.sin(r) * np.cos(phi)
            # ingoing E in y-pol
            self.E[:, :, 1, 1] = (np.cos(r) - 1) * np.sin(phi) * np.cos(phi)
            self.E[:, :, 1, 2] = np.cos(phi) ** 2 + np.cos(r) * np.sin(phi) ** 2
            self.E[:, :, 1, 3] = -np.sin(r) * np.cos(phi)
            self.E = RayBundle.MiV(field_bias, self.E)
        else:
            self.E = None

    def intersect_surface(self, Trf, Rinv=0.0, K=0.0, update=True):
        """
        Gets intersection of a ray bundle and a conic section surface.

        Updates the path with intersection information (unless `update` is set to False, in which case just
        the intersection geometry is returned but the ray positions don't update to the intersection surface).

        Parameters
        ----------
        Trf : np.ndarray of float
            The 4x4 transformation matrix for locating the surface.
        Rinv : float, optional
            The inverse radius of curvature of the surface; 0 = flat (default), positive = curves toward
            object +Z, negative = curves toward object -Z.
        K : float, optional
            Conic constant (-K = eccentricity^2), 0 = spherical (default), -1 = parabola.
        update : bool, optional
            If set to False, only check where the rays hit the surface.

        Returns
        -------
        pos_object : np.ndarray of float
            Object coordinates of where the rays hit the surface; shape (`N`, `N`, 2)
            (so only object x and y coordinates are returned).
        dir_global : np.ndarray of float
            Global unit normal vector of the surface; shape (`N`, `N`, 4).
            Since this is a direction, ``dir_global[:, :, 0] == 0`, and then the 1, 2, and 3 components
            correspond to the x, y, and z directions.
        L : np.ndarray of float
            2D array of the path lengths traversed to reach the surface.

        """

        # rotate to surface coordinates
        x_ = RayBundle.MiV(Trf, self.x)
        p_ = RayBundle.MiV(Trf, self.p)

        # intersection with surface f = 0
        # in the surface coordinates: f = (x**2 + y**2 + (1+K)*z**2)/(2R) - z
        # as a function of distance L traveled: f = aL**2 + bL + c
        c = 0.5 * Rinv * (x_[:, :, 1] ** 2 + x_[:, :, 2] ** 2 + (1 + K) * x_[:, :, 3] ** 2) - x_[:, :, 3]
        b = (
            Rinv
            * (x_[:, :, 1] * p_[:, :, 1] + x_[:, :, 2] * p_[:, :, 2] + (1 + K) * x_[:, :, 3] * p_[:, :, 3])
            - p_[:, :, 3]
        )
        a = 0.5 * Rinv * (p_[:, :, 1] ** 2 + p_[:, :, 2] ** 2 + (1 + K) * p_[:, :, 3] ** 2)

        # now want to solve the quadratic equation, but in the 'stable' sense when a could be zero.
        S = np.where(a * c >= 0, -np.sign(b), np.sign(c))
        discriminant = b**2 - 4 * a * c
        discriminant = np.clip(discriminant, 0, None)  # avoid numerical issues with negative discriminant
        L = 2 * c / (-b + S * np.sqrt(discriminant))

        xs_ = x_ + L[:, :, None] * p_
        if update:
            self.s = self.s + L * self.n_loc

        # get surface normal
        norm = np.zeros((self.N1, self.N2, 4))
        norm[:, :, 0] = 0.0
        norm[:, :, 1] = Rinv * xs_[:, :, 1]
        norm[:, :, 2] = Rinv * xs_[:, :, 2]
        norm[:, :, 3] = Rinv * (1 + K) * xs_[:, :, 3] - 1.0
        d = np.sqrt(np.sum(norm[:, :, 1:] ** 2, axis=-1))
        norm[:, :, 1:] = norm[:, :, 1:] / d[:, :, None]

        # return to standard coords
        if update:
            self.x = self.x + L[:, :, None] * self.p
            # also update trace_history if save_trace_history set to True
            if self.save_trace_history:
                self.trace_history.append(self.x[:,:,1:4].copy())

        pos_object = xs_[:, :, 1:3]
        dir_global = RayBundle.MV(Trf, norm)
        return pos_object, dir_global, L

    def mask(self, Trf, Rinv, K, R, masklist):
        """
        Masks incoming rays at a given surface.

        Parameters
        ----------
        Trf : np.ndarray of float
            The 4x4 transformation matrix for the surface.
        Rinv : float
            The inverse radius of curvature of the surface.
        K : float
            The conic constant for the surface.
        R : float or None
            If not None, mask rays whose intersection points are outside radius `R`.
        masklist : list of dict
            A list of obstructions on the surface. The attributes are:

                ``CIR``, ``REX``, ``REY`` (CODE V codes for shapes)
                ``ADX``, ``ADY``, ``ARO`` (CODE V de-centers)

        Returns
        -------
        None

        Notes
        -----
        All dimensions are in millimeters.

        """

        # get where these rays intersect the surface
        xy_, _, _ = self.intersect_surface(Trf, Rinv=Rinv, K=K, update=False)
        x_ = xy_[:, :, 0]
        y_ = xy_[:, :, 1]
        del xy_

        # outer barrier
        if R is not None:
            self.open = np.where(x_**2 + y_**2 > R**2, False, self.open)
        # inner barriers
        for barrier in masklist:
            Keys = barrier.keys()

            # de-center/rotate if need be
            xl = np.copy(x_)
            yl = np.copy(y_)
            if "ADX" in Keys:
                xl -= barrier["ADX"]
            if "ADY" in Keys:
                yl -= barrier["ADY"]
            if "ARO" in Keys:
                theta = barrier["ARO"] * np.pi / 180.0
                temp_ = np.copy(xl)
                xl = np.cos(theta) * xl + np.sin(theta) * yl
                yl = np.cos(theta) * yl - np.sin(theta) * temp_
                del temp_

            # now do the masking
            if "CIR" in Keys:
                hol = 0.0
                if "HOL" in Keys:
                    hol = barrier["HOL"]
                self.open = np.where(
                    np.logical_and(xl**2 + yl**2 < barrier["CIR"] ** 2, xl**2 + yl**2 >= hol**2),
                    False,
                    self.open,
                )
            if "REX" in Keys:
                # must come with REY
                rect = np.logical_and(np.abs(xl) < barrier["REX"], np.abs(yl) < barrier["REY"])
                if "iCIR_ORIG" in Keys:
                    self.open = np.where(
                        np.logical_and(rect, x_**2 + y_**2 < barrier["iCIR_ORIG"] ** 2), False, self.open
                    )
                else:
                    self.open = np.where(rect, False, self.open)
            if "iREX" in Keys:
                # inversion of REX, used for rectangular apertures: must come with iREY
                # (this isn't a CODE V keyword, but is the easiest way to include the logic here.)
                self.open = np.where(
                    np.logical_and(np.abs(xl) < barrier["iREX"], np.abs(yl) < barrier["iREY"]),
                    self.open,
                    False,
                )

    def intersect_surface_and_reflect(self, Trf, Rinv=0.0, K=0.0, rCoefs=None, activeZone=None, gd=None):
        """
        Propagates rays to a surface and performs a reflection.

        Paramters
        ---------
        Trf : np.ndarray of float
            The 4x4 transformation matrix for the surface.
        Rinv : float, optional
            The inverse radius of curvature of the surface. Default = plane.
        K : float, optional
            The conic constant for the surface. Default = sphere.
        rCoefs : function, optional
            If given, this is a function that takes in an array of incidence angles (1st argument)
            and a vacuum wavelength (2nd argument) and returns S- and P-polarized reflection
            coefficient arrays (complex: same shape as angle of incidence).
            Otherwise assumes a perfect reflecting condition.
        activeZone : dict, optional
            A dictionary of CODEV codes for the active region. Valid keys are
            `CIR``, ``REX``, ``REY``, ``ADX``, ``ADY``, and ``ARO``.
        gd : dict, optional
            A dictionary of parameters for gradient computation. Should have keys:

            - ``grad``: where to write the gradient map;
            - ``surface``: which surface (e.g., ``M1``);
            - ``basis_set``: the RomanBasisSet mapping.

        Returns
        -------
        None

        See Also
        --------
        intersect_surface
            This function differs in that it also performs the reflection.

        """

        # get intersection point
        xy, norm, _ = self.intersect_surface(Trf, Rinv=Rinv, K=K)

        # active zone mask
        if activeZone is not None:
            isGood = np.zeros_like(self.open)
            for z in activeZone:
                Keys = z.keys()
                xl = np.copy(xy[:, :, 0])
                yl = np.copy(xy[:, :, 1])
                if "ADX" in Keys:
                    xl -= z["ADX"]
                if "ADY" in Keys:
                    yl -= z["ADY"]
                if "ARO" in Keys:
                    theta = z["ARO"] * np.pi / 180.0
                    temp_ = np.copy(xl)
                    xl = np.cos(theta) * xl + np.sin(theta) * yl
                    yl = np.cos(theta) * yl - np.sin(theta) * temp_
                    del temp_
                if "CIR" in Keys:
                    obs = 0.0
                    if "OBS" in Keys:
                        obs = z["OBS"]
                    isGood |= np.logical_and(xl**2 + yl**2 < z["CIR"] ** 2, xl**2 + yl**2 >= obs**2)
                if "REX" in Keys:
                    isGood |= np.logical_and(np.abs(xl) < z["REX"], np.abs(yl) < z["REY"])
            self.open &= isGood

        # now let's get the reflected direction
        mu = np.sum(norm * self.p, axis=-1)
        mu_ = np.abs(mu)
        theta_inc = np.where(mu_ < 1, np.arccos(mu_), 0)  # in radians

        # gradient with respect to surface error
        if gd is not None and gd["surface"] in gd["basis_set"].basis:
            b = gd["basis_set"].basis[gd["surface"]]
            modes = b.basis(xy[:, :, 0], xy[:, :, 1])
            for j in range(b.N):
                if "grad" in gd and gd["grad"] is not None:
                    gd["grad"][:, :, b.start + j] += 2 * mu_ * modes[:, :, j]
                if "arr" in gd and gd["arr"] is not None:
                    self.s += 2 * mu_ * modes[:, :, j] * gd["arr"][b.start + j]
            del modes  # save some space

        # flip direction
        p_out = self.p - 2 * mu[:, :, None] * norm

        # if there is an electric field
        if self.E is not None:
            if rCoefs is None:
                RS = RP = -np.ones((self.N1, self.N2), dtype=np.complex128)
            else:
                RS, RP = rCoefs(theta_inc, self.wl)

            # S-type direction as a 3D vector
            Sdir = np.cross(norm[:, :, 1:], self.p[:, :, 1:])
            snorm = np.sum(np.abs(Sdir**2), axis=-1) ** 0.5
            phinorm = np.arctan2(norm[:, :, 2], norm[:, :, 1])
            fallback = np.stack(
                [-np.sin(phinorm), np.cos(phinorm), np.zeros_like(phinorm)],
                axis=-1,
            )
            Sdir_unit = np.empty_like(Sdir, dtype=Sdir.dtype)
            Sdir_unit[...] = fallback
            mask = snorm >= NORM_TOL
            # Normalize Sdir safely; avoid division by very small snorm and ensure correct shape
            np.divide(Sdir, snorm[:, :, None], out=Sdir_unit, where=mask[:, :, None])
            Sdir = Sdir_unit
            del snorm

            # P-type directions
            Pdir_in = np.cross(Sdir, self.p[:, :, 1:])
            Pdir_out = np.cross(p_out[:, :, 1:], Sdir)

            # E-field transformation
            tempS = RS[:, :, None] * np.sum(self.E[:, :, :, 1:] * Sdir[:, :, None, :], axis=-1)
            tempP = RP[:, :, None] * np.sum(self.E[:, :, :, 1:] * Pdir_in[:, :, None, :], axis=-1)
            self.E[:, :, :, 1:] = (
                tempS[:, :, :, None] * Sdir[:, :, None, :] + tempP[:, :, :, None] * Pdir_out[:, :, None, :]
            )
            del tempS, tempP

        # update outgoing direction
        self.p = p_out

    def intersect_surface_and_refract(
        self, Trf, Rinv=0.0, K=0.0, n_new=1.0, tCoefs=None, activeZone=None, gd=None
    ):
        """
        Propagates rays to a surface and performs a refraction.

        Paramters
        ---------
        Trf : np.ndarray of float
            The 4x4 transformation matrix for the surface.
        Rinv : float, optional
            The inverse radius of curvature of the surface. Default = plane.
        K : float, optional
            The conic constant for the surface. Default = sphere.
        n_new : float, optional
            The index of refraction of the new medium. Default = vacuum.
        tCoefs : function, optional
            If given, this is a function that takes in an array of incidence angles (1st argument)
            and a vacuum wavelength (2nd argument) and returns S- and P-polarized reflection
            coefficient arrays (complex: same shape as angle of incidence).
            Otherwise assumes a perfect reflecting condition.
        activeZone : dict, optional
            A dictionary of CODEV codes for the active region. Valid keys are
            `CIR``, ``REX``, ``REY``, ``ADX``, ``ADY``, and ``ARO``.
        gd : dict, optional
            A dictionary of parameters for gradient computation. Should have keys:

            - ``grad``: where to write the gradient map;
            - ``surface``: which surface (e.g., ``M1``);
            - ``basis_set``: the RomanBasisSet mapping.

        Returns
        -------
        None

        See Also
        --------
        intersect_surface
            This function differs in that it also performs the refraction.

        """

        # get intersection point
        xy, norm, _ = self.intersect_surface(Trf, Rinv=Rinv, K=K)

        # active zone mask
        if activeZone is not None:
            isGood = np.zeros_like(self.open)
            for z in activeZone:
                Keys = z.keys()
                xl = np.copy(xy[:, :, 0])
                yl = np.copy(xy[:, :, 1])
                if "ADX" in Keys:
                    xl -= z["ADX"]
                if "ADY" in Keys:
                    yl -= z["ADY"]
                if "ARO" in Keys:
                    theta = z["ARO"] * np.pi / 180.0
                    temp_ = np.copy(xl)
                    xl = np.cos(theta) * xl + np.sin(theta) * yl
                    yl = np.cos(theta) * yl - np.sin(theta) * temp_
                    del temp_
                if "CIR" in Keys:
                    obs = 0.0
                    if "OBS" in Keys:
                        obs = z["OBS"]
                    isGood |= np.logical_and(xl**2 + yl**2 < z["CIR"] ** 2, xl**2 + yl**2 >= obs**2)
                if "REX" in Keys:
                    isGood |= np.logical_and(np.abs(xl) < z["REX"], np.abs(yl) < z["REY"])
            self.open &= isGood

        # now let's get the normal direction
        mu = np.sum(norm * self.p, axis=-1)
        mu_ = np.abs(mu)
        theta_inc = np.where(mu_ < 1, np.arccos(mu_), 0)  # in radians

        # gradient with respect to surface error
        if gd is not None and gd["surface"] in gd["basis_set"].basis:
            dncostheta = self.n_loc * mu_ - np.sqrt(n_new**2 - self.n_loc**2 * np.sin(theta_inc) ** 2)
            b = gd["basis_set"].basis[gd["surface"]]
            modes = b.basis(xy[:, :, 0], xy[:, :, 1])
            for j in range(b.N):
                if "grad" in gd and gd["grad"] is not None:
                    gd["grad"][:, :, b.start + j] += dncostheta * modes[:, :, j] * 8.0 * fratio_scale**2
                if "arr" in gd and gd["arr"] is not None:
                    self.s += dncostheta * modes[:, :, j] * 8.0 * fratio_scale**2 * gd["arr"][b.start + j]
            del modes  # save some space

        # new direction
        n_new__n_old = n_new / self.n_loc
        self.n_loc = n_new
        # Snell's law for new direction
        p_out = (
            self.p
            + (
                (np.sqrt(n_new__n_old**2 - np.sin(theta_inc) ** 2) - np.cos(theta_inc))
                * np.where(mu > 0, 1, -1)
            )[:, :, None]
            * norm
        )
        pnorm = np.sum(np.abs(p_out**2), axis=-1) ** 0.5
        p_out = p_out / pnorm[:, :, None]
        del pnorm

        # if there is an electric field
        if self.E is not None:
            if tCoefs is None:
                TS = TP = -np.ones((self.N1, self.N2), dtype=np.complex128)
            else:
                TS, TP = tCoefs(theta_inc, self.wl)

            # S-type direction as a 3D vector
            Sdir = np.cross(norm[:, :, 1:], self.p[:, :, 1:])
            snorm = np.sum(np.abs(Sdir**2), axis=-1) ** 0.5
            phinorm = np.arctan2(norm[:, :, 2], norm[:, :, 1])
            fallback = np.stack(
                [-np.sin(phinorm), np.cos(phinorm), np.zeros_like(phinorm)],
                axis=-1,
            )
            Sdir_unit = np.empty_like(Sdir, dtype=Sdir.dtype)
            Sdir_unit[...] = fallback
            mask = snorm >= NORM_TOL
            # Normalize Sdir safely; avoid division by very small snorm and ensure correct shape
            np.divide(Sdir, snorm[:, :, None], out=Sdir_unit, where=mask[:, :, None])
            Sdir = Sdir_unit
            del snorm

            # P-type directions
            Pdir_in = np.cross(Sdir, self.p[:, :, 1:])
            Pdir_out = np.cross(p_out[:, :, 1:], Sdir)

            # E-field transformation
            tempS = TS[:, :, None] * np.sum(self.E[:, :, :, 1:] * Sdir[:, :, None, :], axis=-1)
            tempP = TP[:, :, None] * np.sum(self.E[:, :, :, 1:] * Pdir_in[:, :, None, :], axis=-1)
            self.E[:, :, :, 1:] = (
                tempS[:, :, :, None] * Sdir[:, :, None, :] + tempP[:, :, :, None] * Pdir_out[:, :, None, :]
            )
            del tempS, tempP

        # update outgoing direction
        self.p = p_out

    def pad(self, target_size):
        """
        Pads the ray bundle to a target size, centering the current data.

        Parameters
        ----------
        target_size : int
            The desired size of the output arrays (target_size x target_size).

        Returns
        -------
        RayBundle
            A new RayBundle object with padded arrays.

        """
        current_size = self.N1
        if current_size >= target_size:
            raise ValueError("Please increase oversampling factor - can't draw undersampled PSF.")

        # Center the current data in the output arrays
        offset_y = (target_size - current_size) // 2
        offset_x = (target_size - current_size) // 2

        # Create a new RayBundle with target size
        padded_rb = RayBundle.__new__(RayBundle)

        # Copy scalar attributes
        padded_rb.N = target_size
        padded_rb.N1 = target_size
        padded_rb.N2 = target_size
        padded_rb.xan = self.xan
        padded_rb.yan = self.yan
        padded_rb.n_loc = self.n_loc
        padded_rb.costhetaent = self.costhetaent
        padded_rb.wl = self.wl
        padded_rb.wlref = self.wlref

        # Create padded arrays
        padded_rb.x = np.zeros((target_size, target_size, 4))
        padded_rb.p = np.zeros((target_size, target_size, 4))
        padded_rb.open = np.zeros((target_size, target_size), dtype=self.open.dtype)
        padded_rb.s = np.zeros((target_size, target_size), dtype=np.float64)
        padded_rb.xyi = np.zeros((target_size, target_size, 2), dtype=np.float64)
        # NEED TO CHECK THIS!!!
        padded_rb.u = np.full((target_size, target_size, 2), np.nan, dtype=np.float64)

        # Place the data
        padded_rb.open[offset_y : offset_y + current_size, offset_x : offset_x + current_size] = self.open
        padded_rb.x[offset_y : offset_y + current_size, offset_x : offset_x + current_size, :] = self.x
        padded_rb.p[offset_y : offset_y + current_size, offset_x : offset_x + current_size, :] = self.p
        padded_rb.s[offset_y : offset_y + current_size, offset_x : offset_x + current_size] = self.s
        padded_rb.xyi[offset_y : offset_y + current_size, offset_x : offset_x + current_size, :] = self.xyi
        padded_rb.u[offset_y : offset_y + current_size, offset_x : offset_x + current_size, :] = self.u

        # Handle electric field if present
        if self.E is not None:
            padded_rb.E = np.zeros((target_size, target_size, 2, 4), dtype=np.complex128)
            padded_rb.E[offset_y : offset_y + current_size, offset_x : offset_x + current_size, :, :] = self.E
        else:
            padded_rb.E = None

        # Copy x_out if it exists
        if hasattr(self, "x_out"):
            padded_rb.x_out = self.x_out

        return padded_rb

    def fit(self, mode):
        """
        Does a 2D linear fit to the bundle from a given field point.

        Parameters
        ----------
        mode : str
            Which type of fit to do. Options are "xyFPA_from_u" and "u_from_xyi".

        Returns
        -------
        dict
            Keys are ``"Slope"`` (np.ndarray, shape (2, 2)) and ``"Intercept"`` (np.ndarray, shape (2,)).
            The fit is ``output ~ Slope @ input + Intercept``.

        """

        # xyFPA from u:
        if mode.lower() == "xyfpa_from_u":
            if not hasattr(self, "xyFPA"):
                raise AttributeError("This mode requires xyFPA to be set. Use savexy=True.")
            return xyFPA_from_u(self, 0.5)

        # u from xyi:
        if mode.lower() == "u_from_xyi":
            return u_from_xyi(self, 0.5)

        # fallback
        raise ValueError("Unrecognized mode.")


def _RomanRayBundle(
    xan,
    yan,
    N,
    usefilter,
    wl=None,
    hasE=False,
    width=2500.0,
    jacobian=None,
    hires=None,
    ovsamp=6,
    ghostpath=False,
    idealgeom=False,
    idealmirror=False,
    outsca=None,
    errs=None,
    savexy=False,
    save_trace_history=False,
):
    """
    Carries out trace through RST optics.

    Parameters
    ----------
    xan, yan : float
        Angles in degrees in WFI local field angles.
    N : int
        Pupil sampling (NxN grid).
    usefilter : char
        The RST filter element. One of 'R', 'Z', 'Y', 'J', 'H', 'F', 'K', 'W'.
    wl : float, optional
        The wavelength in mm. Default is the central wavelength of that filter.
    hasE : bool, optional
        If True, propagate the electric field.
    width : float, optional
        The size of the grid for sampling of the entrance pupil, in mm. (Default is good for RST.)
    jacobian : np.ndarray, optional
        If used, this will give a 2x2 distortion matrix, used so that the output exit pupil is
        on a square grid. Default is a square grid on the entrance pupil.
    hires : list of np.ndarray of int, optional
        If given, then instead of the full grid, trace only the cells given;
        ``hires[0]`` is a 1D array of y-values and ``hires[1]`` is a 1D array of x-values.
    ovsamp : int, optional
        Oversamples cells in the entrance pupil by this factor; only used if `hires` is given.
    ghostpath : bool, optional
        Whether to include ghost from filter reflections between S1/S2. Default is False.
    idealgeom : bool, optional
        Forces the design model rather than with the best-fit offsets.
    idealmirror : bool, optional
        Forces the mirror to be an ideal conducting surface instead of the model.
    outsca : int, optional
        Which output SCA to project the PSF on to? (1..18)
        The default is to choose the SCA whose center is closest to where the ray lands.
    errs : dict, optional
        The surface error model. Dictionary with keys:

        - basis: ``RomanBasisSet`` object.

        - grad : bool, optional: whether to compute gradient of s with respect to the surface error
          parameters.

          Warning: this can be big!

        - arr: np.ndarray, optional: 1D array of amplitudes of each basis mode.
    savexy : bool, optional
        Whether to save the final xy coordinates of the rays on the FPA.

    Returns
    -------
    psfsim.romantrace.RayBundle
        The ray bundle object. The following additional information is provided as numpy arrays:

            RB.x_out : shape (2,), location of output ray on FPA [in mm]
            RB.xyi : shape (N,N,2), coordinates of the initial rays [in mm]
            RB.u : shape (N,N,2), directions (orthographic projection)
            RB.s : shape (N,N), optical path length of ray to position s

        If `hasE` is True, then also provides:

            RB.E : shape (N,N,2,4), complex, electric field for the 2 initial polarizations and 3 components
           (last axis 0th component should be 0)

        If ``errs["grad"]`` is True, then also provides:
            RB.grad : shape (N,N,errs["basis"].N), derivative of s with respect to each surface error
            parameter.
        If ''savexy'' is True, then also provides:
            RB.xyFPA : shape (N,N,2), the x and y coordinates of the rays on the FPA in mm.

    """

    if jacobian is None:
        jacobian = np.array([[1, 0], [0, 1]])

    wlref = {
        "R": 0.00062,
        "Z": 0.00087,
        "Y": 0.00106,
        "J": 0.00129,
        "H": 0.00158,
        "F": 0.00184,
        "K": 0.00213,
        "W": 0.00146,
    }[usefilter[0].upper()]
    if wl is None:
        wl = wlref

    _reflect_RB_model = None if idealmirror else reflect_RB_model

    # initialization
    RB = RayBundle(
        xan,
        yan,
        N,
        wl=wl,
        wlref=wlref,
        hasE=hasE,
        width=width,
        jacobian=jacobian,
        hires=hires,
        ovsamp=ovsamp,
        idealgeom=idealgeom,
        save_trace_history=save_trace_history,
    )
    if errs is not None and not isinstance(errs, dict):
        raise ValueError("errs must be a dictionary or None")
    grad = errs["grad"] if errs is not None and "grad" in errs else False
    RB.grad = np.zeros(np.shape(RB.s) + (errs["basis"].N,)) if grad else None

    # set up the array of coefficients if we need it
    arr = None
    if errs is not None and "arr" in errs:
        arr = errs["arr"]
        grad = True

    # obstructions:

    # This does nothing except during tests.
    _shadow(RB.open)

    # secondary mirror support tubes
    RB.mask(
        build_transform_matrix(
            xde=-646.3906734739937,
            yde=-392.1337102549381,
            zde=2096.85,
            ade=122.741,
            bde=-37.588,
            cde=-158.584,
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )
    RB.mask(
        build_transform_matrix(
            xde=-646.3906734739937, yde=392.1337102549381, zde=2096.85, ade=-122.741, bde=-37.588, cde=-21.416
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )
    RB.mask(
        build_transform_matrix(
            xde=-16.40275237234653, yde=755.8576220577872, zde=2096.85, ade=-116.448, bde=-15.797, cde=-7.712
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )
    RB.mask(
        build_transform_matrix(
            xde=662.793160207391, yde=363.7239698975893, zde=2096.85, ade=-155.534, bde=61.911, cde=62.718
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )
    RB.mask(
        build_transform_matrix(
            xde=662.793160207391, yde=-363.7239698975893, zde=2096.85, ade=155.534, bde=61.911, cde=117.282
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )
    RB.mask(
        build_transform_matrix(
            xde=-16.40275237234653,
            yde=-755.8576220577872,
            zde=2096.85,
            ade=116.448,
            bde=-15.797,
            cde=-172.288,
        ),
        0,
        0,
        None,
        [{"REX": 38.0, "REY": 1140.0}],
    )

    # secondary mirror baffles
    RB.mask(
        build_transform_matrix(zde=2892.83, ade=-180, cde=180),
        0,
        0,
        None,
        [{"CIR": 358.0, "ADX": 15.56, "HOL": 300.0}],
    )
    RB.mask(
        build_transform_matrix(zde=2892.83),
        0,
        0,
        None,
        [
            {"CIR": 358.0},
            {
                "REX": 250.0,
                "REY": 84.665,
                "ADX": 125.0,
                "ADY": 216.5063509,
                "ARO": -120.0,
                "iCIR_ORIG": 398.02,
            },
        ],
    )
    RB.mask(
        build_transform_matrix(zde=2892.83),
        0,
        0,
        None,
        [{"CIR": 358.0}, {"REX": 250.0, "REY": 84.665, "ADX": -250.0, "iCIR_ORIG": 398.02}],
    )
    RB.mask(
        build_transform_matrix(zde=2892.83),
        0,
        0,
        None,
        [
            {"CIR": 358.0},
            {
                "REX": 250.0,
                "REY": 84.664,
                "ADX": 125.0,
                "ADY": -216.5063509,
                "ARO": 120.0,
                "iCIR_ORIG": 398.02,
            },
        ],
    )

    # primary baffles
    RB.mask(
        build_transform_matrix(zde=697.13, ade=-180, cde=180), 0, 0, None, [{"CIR": 352.3}]
    )  # there is a hole, but we don't account for it at this step
    RB.mask(
        build_transform_matrix(zde=799.518, ade=-180, cde=150), 0, 0, None, [{"CIR": 1e18, "HOL": 1181.56}]
    )  # flipped Boolean from original file since this is a stop

    # now the primary mirror
    RB.intersect_surface_and_reflect(
        build_transform_matrix(zde=660.4, ade=-180, cde=180),
        Rinv=-1.0 / 5671.1342,
        K=-0.9728630311,
        activeZone=[{"CIR": 1184.02, "OBS": 321.31}],
        rCoefs=_reflect_RB_model,
        gd={"grad": RB.grad, "arr": arr, "surface": "M1", "basis_set": errs["basis"]} if grad else None,
    )

    # Secondary mirror
    sm_offset_use = 0.0 if idealgeom else sm_offset["DZ"]
    RB.intersect_surface_and_reflect(
        build_transform_matrix(zde=2945.4 + sm_offset_use, ade=-180, cde=180),
        Rinv=-1.0 / 1299.6164,
        K=-1.6338521231,
        activeZone=[{"CIR": 266.255}],
        rCoefs=_reflect_RB_model,
        gd={"grad": RB.grad, "arr": arr, "surface": "M2", "basis_set": errs["basis"]} if grad else None,
    )

    # PM hole
    RB.mask(
        build_transform_matrix(zde=660.4, ade=-180, cde=180),
        -1.0 / 5671.1342,
        -0.9728630311,
        None,
        [{"CIR": 1e18, "HOL": 321.31}],
    )

    # Fold mirror #1
    RB.intersect_surface_and_reflect(
        build_transform_matrix(
            xde=-73.371025,
            yde=127.0823431034063,
            zde=-299.6,
            ade=135.6742566218209,
            bde=21.96993018862709,
            cde=159.0416363940703,
        ),
        Rinv=0.0,
        K=0.0,
        activeZone=[
            {"REX": 134.13, "REY": 152.42, "ADY": 28.84},
            {"REX": 151.11, "REY": 135.44, "ADY": 28.84},
            {"CIR": 16.98, "ADX": -134.13, "ADY": 164.28},
            {"CIR": 16.98, "ADX": 134.13, "ADY": 164.28},
            {"CIR": 16.98, "ADX": 134.13, "ADY": -106.6},
            {"CIR": 16.98, "ADX": -134.13, "ADY": -106.6},
        ],
        rCoefs=_reflect_RB_model,
        gd={"grad": RB.grad, "arr": arr, "surface": "FM1", "basis_set": errs["basis"]} if grad else None,
    )

    # Entrance aperture plate
    RB.mask(
        build_transform_matrix(
            xde=87.50199,
            yde=-151.5579,
            zde=-325.713,
            ade=99.2177489242795,
            bde=29.6785891029215,
            cde=175.4060606593105,
        ),
        0,
        0,
        None,
        [{"iREX": 146.45, "iREY": 94.98, "ADY": 17.45}],
    )

    # Fold mirror #2
    TF2 = build_transform_matrix(
        xde=466.3656874216886,
        yde=-807.7690655211503,
        zde=-387.2147203297053,
        ade=100.0982115641859,
        bde=29.61417435444774,
        cde=174.9705371700583,
    )
    RB.mask(TF2, 0, 0, None, [{"CIR": 104.0, "ADY": -197.82}])
    RB.intersect_surface_and_reflect(
        TF2,
        Rinv=0.0,
        K=0.0,
        activeZone=[
            {"CIR": 47.7, "ADX": 169.24, "ADY": -92.27},
            {"CIR": 47.7, "ADX": -169.24, "ADY": -92.27},
            {"REX": 169.255, "REY": 47.7, "ADY": -92.27},
            {"REX": 216.955, "REY": 142.135, "ADY": 49.865},
        ],
        rCoefs=_reflect_RB_model,
        gd={"grad": RB.grad, "arr": arr, "surface": "FM2", "basis_set": errs["basis"]} if grad else None,
    )

    # Tertiary mirror
    RB.intersect_surface_and_reflect(
        build_transform_matrix(
            xde=85.50941274300123,
            yde=-148.1066473962562,
            zde=-938.9487447474822,
            ade=117.6411883903639,
            bde=27.08781188751067,
            cde=166.5871249774839,
        ),
        Rinv=-1.0 / 1643.2784,
        K=-0.5965290831,
        activeZone=[
            {"REX": 197.435, "REY": 207.00565, "ADY": 222.29065},
            {"REX": 302.715, "REY": 101.72565, "ADY": 222.29065},
            {"CIR": 105.28, "ADX": -197.435, "ADY": 324.0163},
            {"CIR": 105.28, "ADX": 197.435, "ADY": 324.0163},
            {"CIR": 105.28, "ADX": 197.435, "ADY": 120.565},
            {"CIR": 105.28, "ADX": -197.435, "ADY": 120.565},
            {"REX": 256.189698, "REY": 31.9859, "ADY": 444.7891},
        ],
        rCoefs=_reflect_RB_model,
        gd={"grad": RB.grad, "arr": arr, "surface": "M3", "basis_set": errs["basis"]} if grad else None,
    )

    # Exit pupil mask
    PupilLoc = build_transform_matrix(
        xde=526.6061126910644,
        yde=-912.1085427572655,
        zde=-541.0972086432624,
        ade=91.27416530783904,
        bde=29.99386508559931,
        cde=179.3629567305957,
    )
    xy, _, _ = RB.intersect_surface(PupilLoc, update=False)
    if usefilter[0].upper() in ["F", "K"]:
        RB.mask(
            PupilLoc,
            0,
            0,
            44.3,
            [
                {"CIR": 17.1, "ADY": -0.5},
                {"REY": 1.5, "REX": 22.2, "ADX": -30.8563, "ADY": 1.2268, "ARO": 160.5},
                {"REY": 1.5, "REX": 22.2, "ADX": -13.9795, "ADY": -26.6518, "ARO": -103},
                {"REY": 1.5, "REX": 22.2, "ADX": 13.9795, "ADY": -26.6518, "ARO": -77},
                {"REY": 1.5, "REX": 22.2, "ADX": 30.8563, "ADY": 1.2268, "ARO": 19.5},
                {"REY": 1.5, "REX": 22.2, "ADX": 16.1804, "ADY": 24.9799, "ARO": 44.1},
                {"REY": 1.5, "REX": 22.2, "ADX": -16.1804, "ADY": 24.9799, "ARO": 135.9},
            ],
        )
    else:
        RB.mask(
            PupilLoc,
            0,
            0,
            47.5,
            [
                {"CIR": 12.0, "ADY": -0.5},
                {"REY": 0.5, "REX": 22.2, "ADX": -30.8563, "ADY": 1.2268, "ARO": 160.5},
                {"REY": 0.5, "REX": 22.2, "ADX": -13.9795, "ADY": -26.6518, "ARO": -103},
                {"REY": 0.5, "REX": 22.2, "ADX": 13.9795, "ADY": -26.6518, "ARO": -77},
                {"REY": 0.5, "REX": 22.2, "ADX": 30.8563, "ADY": 1.2268, "ARO": 19.5},
                {"REY": 0.5, "REX": 22.2, "ADX": 16.1804, "ADY": 24.9799, "ARO": 44.1},
                {"REY": 0.5, "REX": 22.2, "ADX": -16.1804, "ADY": 24.9799, "ARO": 135.9},
            ],
        )

    # DEFINE/GET STUFF FOR S1
    S1 = build_transform_matrix(
        xde=531.5125171153529,
        yde=-920.606684502614,
        zde=-539.1713886876893,
        ade=102.76851389522,
        bde=29.38268469198068,
        cde=173.6554980927907,
    )
    Rinv1 = -1.0 / 1500.0
    K1 = 0.0
    n_new1 = n_Infrasil301(wlref)
    activeZone1 = [{"CIR": 52.65}]

    # DEFINE/GET STUFF FOR S2
    S2 = build_transform_matrix(
        xde=536.4189215396419,
        yde=-929.1048262479633,
        zde=-537.2455687321163,
        ade=102.76851389522,
        bde=29.38268469198068,
        cde=173.6554980927907,
    )
    Rinv2 = -1.0 / 1499.31453814
    K2 = 0.0
    n_new2 = 1.0
    activeZone2 = [{"CIR": 52.65}]

    # now for int/refract/reflect including ghost
    # step 1... see photo
    # int/refract through S1
    gd1 = {"grad": RB.grad, "arr": arr, "surface": "S1", "basis_set": errs["basis"]} if grad else None
    RB.intersect_surface_and_refract(S1, Rinv=Rinv1, K=K1, n_new=n_new1, activeZone=activeZone1, gd=gd1)

    # ghost time
    if ghostpath:
        _, _, L = RB.intersect_surface(S2, Rinv=Rinv2, K=K2, update=False)
        RB.s += L * (n_Infrasil301(wl) - n_Infrasil301(wlref))
        RB.intersect_surface_and_reflect(S2, Rinv=Rinv2, K=K2, activeZone=activeZone2)
        _, _, L = RB.intersect_surface(S1, Rinv=Rinv1, K=K1, update=False)
        RB.s += L * (n_Infrasil301(wl) - n_Infrasil301(wlref))
        RB.intersect_surface_and_reflect(S1, Rinv=Rinv1, K=K1, activeZone=activeZone1, gd=gd1)

    # down here is the refract through S2
    _, _, L = RB.intersect_surface(S2, Rinv=Rinv2, K=0.0, update=False)
    RB.s += L * (n_Infrasil301(wl) - n_Infrasil301(wlref))
    RB.intersect_surface_and_refract(S2, Rinv=Rinv2, K=K2, n_new=n_new2, activeZone=activeZone2)

    # FPA
    TrFPA = build_transform_matrix(
        xde=866.9584811995454,
        yde=-1501.61613749036,
        zde=-407.5050041451383,
        ade=-62.41145131632292,
        bde=-27.09897706981732,
        cde=13.3889006733882,
    )
    if not idealgeom:
        TrFPA = TrFPA @ build_transform_matrix(
            xde=-fpa_offset["DX"] - fbias_offset["HORIZONTAL"] * 0.01 / 0.11 * 3600.0,
            yde=-fpa_offset["DY"] + fbias_offset["VERTICAL"] * 0.01 / 0.11 * 3600.0,
            zde=-fpa_offset["DZ"],
            ade=fpa_offset["TILT"],
            bde=fpa_offset["TIP"],
            cde=fpa_offset["ROLL"],
        )

    xyFPA, _, _ = RB.intersect_surface(TrFPA, Rinv=0.0, K=0.0, update=True)
    if savexy:
        RB.xyFPA = xyFPA
    RB.u = np.einsum("ij,abj->abi", np.linalg.inv(TrFPA), RB.p)[:, :, 1:3]
    if hasE:
        RB.E = np.einsum("ij,abkj->abki", np.linalg.inv(TrFPA), RB.E)

    # get position of central ray and update wavefront map accordingly
    if hires is None:
        RB.x_out = np.mean(xyFPA[N // 2 - 1 : N // 2 + 1, N // 2 - 1 : N // 2 + 1, :], axis=(0, 1))
    else:
        # this is robust against having no rays at all
        RB.x_out = np.mean(xyFPA, axis=(0, 1)) if np.shape(xyFPA)[0] else np.zeros((2,))
    RB.s += np.sum(RB.u * (RB.x_out[None, None, :] - xyFPA), axis=-1)

    # now get which chip we want to project onto
    if outsca is None:
        # "smallest bounding square" distance in the focal plane
        dist = np.amax(np.abs(RB.x_out[None, :] - scapos), axis=1)
        outsca = 1 + np.argmin(dist)

    # get gradients with respect to the displacement of each SCA
    if grad:
        key = f"WFI{outsca:02d}"
        if key in errs["basis"].basis:
            b = errs["basis"].basis[key]
            dx_from_ctr = RB.x_out - scapos[outsca - 1]
            dz = b.basis(dx_from_ctr[0], dx_from_ctr[1]) * 8.0 * fratio_scale**2
            w = np.sqrt(1.0 - RB.u[:, :, 0] ** 2 - RB.u[:, :, 1] ** 2)
            for j in range(b.N):
                if RB.grad is not None:
                    RB.grad[:, :, b.start + j] += dz[j] * w
                if arr is not None:
                    RB.s += arr[b.start + j] * dz[j] * w

    return RB


# ELLE DOING STUFF HERE TOO


def RomanRayBundle(
    xan,
    yan,
    N,
    usefilter,
    wl=None,
    hasE=False,
    width=2500.0,
    jacobian=None,
    ovsamp=6,
    a_lanczos=3,
    idealgeom=False,
    idealmirror=False,
    outsca=None,
    errs=None,
    ghostpath=False,
    savexy=False,
):
    """
    Carries out trace through RST optics.

    Parameters
    ----------
    xan, yan : float
        Angles in degrees in WFI local field angles.
    N : int
        Pupil sampling (NxN grid).
    usefilter : char
        The RST filter element. One of 'R', 'Z', 'Y', 'J', 'H', 'F', 'K', 'W'.
    wl : float, optional
        The wavelength in mm. Default is the central wavelength of that filter.
    hasE : bool, optional
        If True, propagate the electric field.
    width : float, optional
        The size of the grid for sampling of the entrance pupil, in mm. (Default is good for RST.)
    jacobian : np.ndarray, optional
        If used, this will give a 2x2 distortion matrix, used so that the output exit pupil is
        on a square grid. Default is a square grid on the entrance pupil.
    ovsamp : int, optional
        Oversamples cells in the entrance pupil by this factor.
    a_lanczos : int, optional
        The "a" parameter for Lanczos interpolation; this controls the size of the kernel. Default is 3.
    idealgeom : bool, optional
        Forces the design model rather than with the best-fit offsets.
    idealmirror : bool, optional
        Forces the mirror to be an ideal conducting surface instead of the model.
    outsca : int, optional
        Which output SCA to project the PSF on to? (1..18)
        The default is to choose the SCA whose center is closest to where the ray lands.
    errs : dict, optional
        The surface error model. Dictionary with keys:

        - basis: ``RomanBasisSet`` object.

        - grad : bool, optional: whether to compute gradient of s with respect to the surface error
          parameters.

          Warning: this can be big!

        - arr: np.ndarray, optional: 1D array of amplitudes of each basis mode.
    ghostpath : bool, optional
        Whether to include ghost from filter reflections between S1/S2. Default is False.
    savexy : bool, optional
        Whether to save the final xy coordinates of the rays on the FPA.

    Returns
    -------
    psfsim.romantrace.RayBundle
        The ray bundle object. The following additional information is provided as numpy arrays:

            RB.x_out : shape (2,), location of output ray on FPA [in mm]
            RB.xyi : shape (N,N,2), coordinates of the initial rays [in mm]
            RB.u : shape (N,N,2), directions (orthographic projection)
            RB.s : shape (N,N), optical path length of ray to position s
            RS.open : shape (N,N), fraction of that cell that is open (between 0 and 1 inclusive).

        If `hasE` is True, then also provides:

            RB.E : shape (N,N,2,4), complex, electric field for the 2 initial polarizations and 3 components
           (last axis 0th component should be 0)

        If ``errs["grad"]`` is True, then also provides:
            RB.grad : shape (N,N,errs["basis"].N), derivative of s with respect to each surface error
            parameter.
        If ''savexy'' is True, then also provides:
            RB.xyFPA : shape (N,N,2), the x and y coordinates of the rays on the FPA in mm.

    See Also
    --------
    _RomanRayBundle
        The routine "wrapped inside" (this is called once to figure out which cells need to be
        simulated at higher resolution).

    Notes
    -----
    This is based on the specifications in:
    "Opto-Mechanical Definitions", RST-SYS-SPEC-0055, Revision E
    released by the Configuration Management Office June 11, 2021
    (not export controlled).
    Most information is in CODE V format, but some was converted to be
    usable in this Python script.

    """

    RB = _RomanRayBundle(
        xan,
        yan,
        N,
        usefilter,
        wl=wl,
        hasE=hasE,
        width=width,
        jacobian=jacobian,
        hires=None,
        idealgeom=idealgeom,
        idealmirror=idealmirror,
        outsca=outsca,
        errs=errs,
        ghostpath=ghostpath,
        savexy=savexy,
    )

    # Now figure out which pixels we need to increase the resolution.
    r = 40.0 / width * N  # radius of search in pixels
    rceil = int(np.ceil(r))
    s = np.linspace(-rceil, rceil, 2 * rceil + 1)
    x_, y_ = np.meshgrid(s, s)
    cfilter = np.where(np.hypot(x_, y_) < r, 1.0, 0.0)
    n = np.sum(cfilter)
    RB.open = RB.open.astype(np.float64)
    _open = scipy.ndimage.convolve(RB.open, cfilter, mode="nearest") / n
    bdycells = np.where(np.logical_and(_open > 0.5 / n, 1 - _open > 0.5 / n))
    del _open

    # run these pixels specifically at higher resolution
    RB_hires = _RomanRayBundle(
        xan,
        yan,
        N,
        usefilter,
        wl=wl,
        hasE=False,
        width=width,
        jacobian=jacobian,
        hires=bdycells,
        ovsamp=ovsamp,
        idealmirror=True,
        ghostpath=ghostpath,
        savexy=False,
    )

    new_values = _apply_lanczos_reweighting(
        RB.open,
        RB_hires.open,
        bdycells,
        ovsamp,
        a_lanczos,
    )

    RB.open[bdycells[0], bdycells[1]] = new_values
    # force to zeros where closed only if RB.E is not None
    if RB.E is not None:
        for i in range(2):
            for j in range(4):
                RB.E[:, :, i, j] = np.where(RB.open > 1e-16, RB.E[:, :, i, j], 0.0)

    return RB


def demo(writefiles=False, savexy=False):
    """
    Demo and test functions for romantrace.

    Parameters
    ----------
    writefiles : bool, optional
        Write the output files?
    savexy : bool, optional
        Save the final (x, y) in the returned RomanRayBundle object?

    Returns
    -------
    RomanRayBundle
        The ray bundle from this run.

    """

    # Transformation matrix test
    Rtrans = build_transform_matrix(
        xde=-73.371025,
        yde=127.082343,
        zde=-299.600000,
        ade=135.674257,
        bde=21.969930,
        cde=159.041636,
        unit="degree",
    )
    print(Rtrans)
    Rtrans_target = np.array(
        [
            [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [-7.33710250e01, -8.66025403e-01, -3.31714146e-01, -3.74119937e-01],
            [1.27082343e02, -5.00000002e-01, 5.74545746e-01, 6.47994740e-01],
            [-2.99600000e02, -3.88564025e-09, 7.48239875e-01, -6.63428285e-01],
        ]
    )
    assert np.all(np.abs(Rtrans - Rtrans_target) < 1e-6)

    # Ray bundle test
    RB = RayBundle(-0.071, -0.037, 2)
    print("--x--")
    print(RB.x)
    print("--p--")
    print(RB.p)
    assert np.all(
        np.abs(
            RB.x
            - np.array(
                [
                    [
                        [1.00000000e00, 2.43926855e02, 8.27506293e02, 3.50527931e03],
                        [1.00000000e00, -8.38604899e02, 2.02506293e02, 3.50527931e03],
                    ],
                    [
                        [1.00000000e00, 8.68903437e02, -2.54984899e02, 3.49445840e03],
                        [1.00000000e00, -2.13628318e02, -8.79984899e02, 3.49445840e03],
                    ],
                ]
            )
        )
        < 1e-4
    )
    assert np.all(
        np.abs(
            RB.p
            - np.array(
                [
                    [
                        [0.0, -0.00293232, 0.00755729, -0.99996714],
                        [0.0, -0.00293232, 0.00755729, -0.99996714],
                    ],
                    [
                        [0.0, -0.00293232, 0.00755729, -0.99996714],
                        [0.0, -0.00293232, 0.00755729, -0.99996714],
                    ],
                ]
            )
        )
        < 1e-5
    )

    # pupils
    RB = RomanRayBundle(
        -0.399, 0.208, 512, "W", wl=9.27e-4, hasE=True, idealgeom=True, idealmirror=True, savexy=savexy
    )
    if writefiles:
        fits.PrimaryHDU(RB.open.astype(np.int8)).writeto("temp.fits", overwrite=True)
        fits.PrimaryHDU(np.where(RB.open, RB.s - np.median(RB.s), 0)).writeto("temp-s.fits", overwrite=True)
        fits.PrimaryHDU(np.where(RB.open, RB.u[:, :, 0], 0)).writeto("temp-u.fits", overwrite=True)
    # pupil test
    frac = np.mean(RB.open)
    print("frac open **", frac)
    assert np.abs(frac - 0.5586506525675455) < 1e-3

    # Electric fields
    print("-- E out --")
    print(RB.E[128, 128, :, :])
    print(RB.x[::64, ::64, 1:])
    print(RB.p[::64, ::64, 1:])
    print(RB.u[::64, ::64, :])
    # test against "correct" answer
    tmp_arr = np.array(
        [
            [0.0 + 0.0j, 0.92894895 + 0.0j, 0.3679539 + 0.0j, 0.04078929 + 0.0j],
            [0.0 + 0.0j, -0.08449756 + 0.0j, 0.10352705 + 0.0j, 0.99104315 + 0.0j],
        ]
    )
    rotFPA = build_transform_matrix(
        ade=-62.41145131632292,
        bde=-27.09897706981732,
        cde=13.3889006733882,
    )  # this is to rotate to the correct answer to the FPA coordinates.
    assert np.all(np.abs(RB.E[128, 128, :, :] - tmp_arr @ rotFPA) < 1e-5)
    out_pos = np.array([766.73306894, -1593.99400015, -473.55384725])
    assert np.all(np.abs(RB.x[::64, ::64, 1:] - out_pos[None, None, :]) < 0.1)
    _n = np.shape(RB.u)[0]
    assert np.all(
        np.abs(
            RB.u[:: _n - 1, :: _n - 1, :]
            - np.array(
                [
                    [[-0.1184445, -0.25782568], [-0.2472246, -0.25452657]],
                    [[-0.11510895, -0.38269087], [-0.243886, -0.37932103]],
                ]
            )
        )
        < 1e-5
    )

    print("-->", RB.x_out)

    print("-- n table --")
    for wl in np.linspace(6e-4, 2.4e-3, 37):
        print(f"{wl:11.5E} {n_Infrasil301(wl):8.6f}")
    assert np.abs(n_Infrasil301(0.8 / 1000.0) - 1.452973) < 1e-4
    assert np.abs(n_Infrasil301(2.1 / 1000.0) - 1.436305) < 1e-4

    E = RB.E * RB.open[:, :, None, None]

    im = np.stack(
        (
            RB.xyi[:, :, 0],
            RB.xyi[:, :, 1],
            RB.u[:, :, 0],
            RB.u[:, :, 1],
            RB.s,
            RB.open.astype(np.float64),
            np.sum(np.abs(E[:, :, 0, :]) ** 2, axis=-1),
            np.sum(np.abs(E[:, :, 1, :]) ** 2, axis=-1),
            E[:, :, 0, 1].real,
            E[:, :, 0, 1].imag,
            E[:, :, 0, 2].real,
            E[:, :, 0, 2].imag,
            E[:, :, 0, 3].real,
            E[:, :, 0, 3].imag,
            E[:, :, 1, 1].real,
            E[:, :, 1, 1].imag,
            E[:, :, 1, 2].real,
            E[:, :, 1, 2].imag,
            E[:, :, 1, 3].real,
            E[:, :, 1, 3].imag,
        )
    )
    if writefiles:
        fits.PrimaryHDU(im).writeto("pupil_diagnostics.fits", overwrite=True)

    return RB
