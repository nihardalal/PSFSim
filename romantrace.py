import numpy as np
from astropy.io import fits

version = '250506'

### begin material data ###

def n_Infrasil301(wl, T=180.):
    """Index of refraction of Infrasil301. wl in mm, T in Kelvin. Both scalars.

    Sellmeier coefficients from Leviton, Frey, & Madison (2008)
    arXiv:0805.0096
    """

    data = np.array([[0.105962, 0.995429, 0.865120, 4.500743E-03, 9.383735E-02, 9.757183],
            [9.359142E-06, -7.973196E-06, 3.731950E-04, -2.825065E-04, -1.374171E-06, 1.864621E-03],
            [4.941067E-08, 1.006343E-09, -2.010347E-06, 3.136868E-06, 1.316037E-08, -1.058414E-05],
            [4.890163E-11, -8.694712E-11, 2.708606E-09, -1.121499E-08, 1.252909E-11, 1.730321E-08],
            [1.492126E-13, -1.220612E-13, 1.679976E-12, 1.236514E-11, -4.641280E-14, 1.719396E-12]])
    D = data.T @ np.array([1,T,T**2,T**3,T**4])
    L = wl*1e3 # convert to microns
    return np.sqrt(1 + np.sum(D[:3]*L**2/(L**2-D[-3:]**2)) )

### end material data ###

def build_transform_matrix(xde=0., yde=0., zde=0., ade=0., bde=0., cde=0., unit='degree'):
    """Makes a 4x4 transformation matrix:
    u(global coords) = R u(obj coords)
    where u is a vector of [1,x,y,z]

    Legal units are 'degree' or 'radian'
    """

    # conversion
    if unit.lower()[:3] == 'deg':
       ade *= np.pi/180.
       bde *= np.pi/180.
       cde *= np.pi/180.

    R = np.zeros((4,4))
    R[0,0] = 1.
    R[1,0] = xde
    R[2,0] = yde
    R[3,0] = zde

    # the [1:,1:] sub-block is the rotation matrix
    R[1:,1:] = \
               np.array([ [1,0,0], [0,np.cos(ade),np.sin(ade)], [0,-np.sin(ade),np.cos(ade)]])\
              @np.array([ [np.cos(bde), 0, -np.sin(bde)], [0,1,0], [np.sin(bde),0,np.cos(bde)]]) \
              @np.array([ [np.cos(cde),-np.sin(cde),0], [np.sin(cde),np.cos(cde),0], [0,0,1]])
    return R

class RayBundle:

    """Class defining a ray bundle. This has a bunch of attributes for 2D array of rays.

    Attributes:
        N : size of ray bundle
        x : position of rays (0th element 1)
        p : direction of rays (0th element 0)
        s : path length
        n_loc : local index of refraction
        xan, yan: field angles (in degrees)
        costhetaent: projection factor for entrance aperture
        open : boolean (has this ray propagated)
        wl, wlref : wavelength and reference wavelength (the latter for geometric trace)
        E : electric field (optional, 4D: 2D for array, 1D for input pol, 1D for output pol) - None if not used
    """

    # forward and reverse transformations
    @classmethod
    def MV(self,a,b):
        return np.einsum('ij,...j->...i', a, b)
    @classmethod
    def MiV(self,a,b):
        a_ = np.linalg.inv(a)
        a_[0,0] = 1.
        a_[0,1:] = 0.
        return np.einsum('ij,...j->...i', a_, b)

    def __init__(self, xan, yan, N, hasE=False, width=2500., startpos = 3500., wl=1.29e-3, wlref=1.29e-3, jacobian = np.array([[1,0],[0,1]])):
        """Constructor, from a given position xan, yan in WFI coordiantes (in degrees)
        and the given bundle size (N x N)
        The 'hasE' argument tells whether to build an E-field or just do the ray trace (False, default)
        Jacobian will read in a 2 by 2 matrix, defaulting to identity. 
        """

        self.N = N
        self.xan = xan; self.yan = yan
        self.x = np.zeros((N,N,4))
        self.p = np.zeros((N,N,4))
        self.open = np.ones((N,N), dtype=bool)
        self.n_loc = 1.

        # make a grid for the entrance pupil
        s = np.linspace(-width/2.*(1-1./N),width/2.*(1-1./N),N)
        xi,yi = np.meshgrid(s,s)
        self.x[:,:,0] = 1.
        self.x[:,:,1] = jacobian[0][0]*xi+jacobian[0][1]*yi
        self.x[:,:,2] = jacobian[1][0]*xi+jacobian[1][1]*yi
        self.x[:,:,3] = startpos

        xi = self.x[:,:, 1]
        yi = self.x[:,:, 2]

        self.xyi = self.x[:,:,1:3]

        # get directions
        r = np.sqrt(xan**2+yan**2)*np.pi/180.
        self.costhetaent = np.cos(r)
        phi = np.arctan2(yan,-xan)
        self.p[:,:,1] = -np.sin(r)*np.cos(phi)
        self.p[:,:,2] = -np.sin(r)*np.sin(phi)
        self.p[:,:,3] = -np.cos(r)

        self.wl = wl
        self.wlref = wlref

        # initial path length
        self.s = xi*self.p[:,:,1] + yi*self.p[:,:,2]

        # remove field bias, rotate to Payload Coordinate System
        field_bias = build_transform_matrix(ade=-0.496, cde=150., unit='degree')
        self.x = RayBundle.MiV(field_bias, self.x)
        self.p = RayBundle.MiV(field_bias, self.p)

        # if requested, build the E-field
        if hasE:
            self.E = np.zeros((N,N,2,4), dtype=np.complex128)
            # ingoing E in x-pol
            self.E[:,:,0,1] = np.sin(phi)**2 + np.cos(r)*np.cos(phi)**2
            self.E[:,:,0,2] = (np.cos(r)-1)*np.sin(phi)*np.cos(phi)
            self.E[:,:,0,3] = -np.sin(r)*np.cos(phi)
            # ingoing E in y-pol
            self.E[:,:,1,1] = (np.cos(r)-1)*np.sin(phi)*np.cos(phi)
            self.E[:,:,1,2] = np.cos(phi)**2 + np.cos(r)*np.sin(phi)**2
            self.E[:,:,1,3] = -np.sin(r)*np.cos(phi)
            self.E = RayBundle.MiV(field_bias, self.E)
        else:
            self.E = None

    def intersect_surface(self, Trf, Rinv=0., K=0., update=True):
        """Gets intersection with a surface with transform matrix Trf and given curvature and conic constant.

        Updates the path with intersection information (unless update is set to False, in which case just
        the intersection geometry is returned but the ray positions don't update to the intersection plane).
        Returns:

        *  intersection locations (N,N,2) in surface coordiantes
        *  normal vector (N,N,4) at the intersection.
        *  normal vector (N,N,2,4) at the intersection.
        *  path length
        """

        # rotate to surface coordinates
        x_ = RayBundle.MiV(Trf,self.x)
        p_ = RayBundle.MiV(Trf,self.p)

        # intersection with surface f = 0
        # in the surface coordinates: f = (x**2 + y**2 + (1+K)*z**2)/(2R) - z
        # as a function of distance L traveled: f = aL**2 + bL + c
        c = .5*Rinv*(x_[:,:,1]**2 + x_[:,:,2]**2 + (1+K)*x_[:,:,3]**2) - x_[:,:,3]
        b = Rinv*(x_[:,:,1]*p_[:,:,1] + x_[:,:,2]*p_[:,:,2] + (1+K)*x_[:,:,3]*p_[:,:,3]) - p_[:,:,3]
        a = .5*Rinv*(p_[:,:,1]**2 + p_[:,:,2]**2 + (1+K)*p_[:,:,3]**2)

        # now want to solve the quadratic equation, but in the 'stable' sense when a could be zero.
        S = np.where(a*c>=0, -np.sign(b), np.sign(c))
        L = 2*c / ( -b + S*np.sqrt(b**2-4*a*c) )

        xs_ = x_ + L[:,:,None]*p_
        if update: self.s = self.s + L*self.n_loc

        # get surface normal
        norm = np.zeros((self.N,self.N,4))
        norm[:,:,0] = 0.
        norm[:,:,1] = Rinv*xs_[:,:,1]
        norm[:,:,2] = Rinv*xs_[:,:,2]
        norm[:,:,3] = Rinv*(1+K)*xs_[:,:,3] -1.
        d = np.sqrt(np.sum(norm[:,:,1:]**2, axis=-1))
        norm[:,:,1:] = norm[:,:,1:] / d[:,:,None]

        # return to standard coords
        if update: self.x = self.x + L[:,:,None]*self.p
        return xs_[:,:,1:3], RayBundle.MV(Trf,norm), L

    def mask(self, Trf, Rinv, K, R, masklist):
        """Masks incoming rays at a given surface.
        The mask has coordinate coordinate transform Trf and radius R (set to None if no outer barrier).
        The masklist is a list of dictionary obstructions, which have attributes:
            CIR, REX, REY (CODE V codes for shapes)
            ADX, ADY, ARO (CODE V de-centers)
        """

        # get where these rays intersect the surface
        xy_, _, _ = self.intersect_surface(Trf, Rinv=Rinv, K=K, update=False)
        x_ = xy_[:,:,0]
        y_ = xy_[:,:,1]
        del xy_

        # outer barrier
        if R is not None:
            self.open = np.where(x_**2+y_**2>R**2, False, self.open)
        # inner barriers
        for barrier in masklist:
            Keys = barrier.keys()

            # de-center/rotate if need be
            xl = np.copy(x_)
            yl = np.copy(y_)
            if 'ADX' in Keys: xl -= barrier['ADX']
            if 'ADY' in Keys: yl -= barrier['ADY']
            if 'ARO' in Keys:
                theta = barrier['ARO'] * np.pi/180.
                temp_ = np.copy(xl)
                xl = np.cos(theta)*xl + np.sin(theta)*yl
                yl = np.cos(theta)*yl - np.sin(theta)*temp_
                del temp_

            # now do the masking
            if 'CIR' in Keys:
                hol = 0.
                if 'HOL' in Keys: hol = barrier['HOL']
                self.open = np.where(np.logical_and(xl**2+yl**2<barrier['CIR']**2, xl**2+yl**2>=hol**2), False, self.open)
            if 'REX' in Keys:
                # must come with REY
                rect = np.logical_and(np.abs(xl)<barrier['REX'], np.abs(yl)<barrier['REY'])
                if 'iCIR_ORIG' in Keys:
                    self.open = np.where(np.logical_and(rect, x_**2+y_**2<barrier['iCIR_ORIG']**2), False, self.open)
                else:
                    self.open = np.where(rect, False, self.open)
            if 'iREX' in Keys:
                # inversion of REX, used for rectangular apertures: must come with iREY
                # (this isn't a CODE V keyword, but is the easiest way to include the logic here.)
                self.open = np.where(np.logical_and(np.abs(xl)<barrier['iREX'], np.abs(yl)<barrier['iREY']), self.open, False)

    def intersect_surface_and_reflect(self, Trf, Rinv=0., K=0., rCoefs=None, activeZone=None):
        """Like intersect_surface, but performs the reflection.

        If rCoefs is given, this is taken to be a function that takes in arrays of incidence angles and
        returns S- and P-polarized reflection coefficient arrays.
        Otherwise assumes a perfect reflecting condition.

        activeZone is a dictionary (CODE V codes), list thereof, or None.
        """

        # get intersection point
        xy, norm, _ = self.intersect_surface(Trf,Rinv=Rinv,K=K)

        # active zone mask
        if activeZone is not None:
            isGood = np.zeros_like(self.open)
            for z in activeZone:
                Keys = z.keys()
                xl = np.copy(xy[:,:,0])
                yl = np.copy(xy[:,:,1])
                if 'ADX' in Keys: xl -= z['ADX']
                if 'ADY' in Keys: yl -= z['ADY']
                if 'ARO' in Keys:
                    theta = z['ARO'] * np.pi/180.
                    temp_ = np.copy(xl)
                    xl = np.cos(theta)*xl + np.sin(theta)*yl
                    yl = np.cos(theta)*yl - np.sin(theta)*temp_
                    del temp_
                if 'CIR' in Keys:
                    obs = 0.
                    if 'OBS' in Keys: obs=z['OBS']
                    isGood |= np.logical_and(xl**2+yl**2<z['CIR']**2, xl**2+yl**2>=obs**2)
                if 'REX' in Keys:
                    isGood |= np.logical_and(np.abs(xl)<z['REX'], np.abs(yl)<z['REY'])
            self.open &= isGood

        # now let's get the reflected direction
        mu = np.sum(norm*self.p, axis=-1)
        mu_ = np.abs(mu)
        theta_inc = np.where(mu_<1, np.arccos(mu_), 0) # in radians

        # flip direction
        p_out = self.p - 2*mu[:,:,None]*norm

        # if there is an electric field
        if self.E is not None:
            if rCoefs is None:
                RS = RP = -np.ones((self.N,self.N), dtype=np.complex128)
            else:
                RS,RP = rCoefs(theta_inc)

            # S-type direction as a 3D vector
            Sdir = np.cross(norm[:,:,1:],self.p[:,:,1:])
            snorm = np.sum(np.abs(Sdir**2),axis=-1)**.5
            Sdir = Sdir/snorm[:,:,None]
            del snorm

            # P-type directions
            Pdir_in = np.cross(Sdir,self.p[:,:,1:])
            Pdir_out = np.cross(p_out[:,:,1:],Sdir)

            # E-field transformation
            tempS = RS[:,:,None] * np.sum(self.E[:,:,:,1:] * Sdir[:,:,None,:], axis=-1)
            tempP = RP[:,:,None] * np.sum(self.E[:,:,:,1:] * Pdir_in[:,:,None,:], axis=-1)
            self.E[:,:,:,1:] = tempS[:,:,:,None] * Sdir[:,:,None,:] + tempP[:,:,:,None] * Pdir_out[:,:,None,:]
            del tempS, tempP

        # update outgoing direction
        self.p=p_out

    def intersect_surface_and_refract(self, Trf, Rinv=0., K=0., n_new=1., tCoefs=None, activeZone=None):
        """Like intersect_surface, but performs a refraction into a new medium with n=n_new.

        If tCoefs is given, this is taken to be a function that takes in arrays of incidence angles and
        returns S- and P-polarized transmission coefficient arrays.
        Otherwise assumes a perfect AR coating.

        activeZone is a dictionary (CODE V codes), list thereof, or None.
        """

        # get intersection point
        xy, norm, _ = self.intersect_surface(Trf,Rinv=Rinv,K=K)

        # active zone mask
        if activeZone is not None:
            isGood = np.zeros_like(self.open)
            for z in activeZone:
                Keys = z.keys()
                xl = np.copy(xy[:,:,0])
                yl = np.copy(xy[:,:,1])
                if 'ADX' in Keys: xl -= z['ADX']
                if 'ADY' in Keys: yl -= z['ADY']
                if 'ARO' in Keys:
                    theta = z['ARO'] * np.pi/180.
                    temp_ = np.copy(xl)
                    xl = np.cos(theta)*xl + np.sin(theta)*yl
                    yl = np.cos(theta)*yl - np.sin(theta)*temp_
                    del temp_
                if 'CIR' in Keys:
                    obs = 0.
                    if 'OBS' in Keys: obs=z['OBS']
                    isGood |= np.logical_and(xl**2+yl**2<z['CIR']**2, xl**2+yl**2>=obs**2)
                if 'REX' in Keys:
                    isGood |= np.logical_and(np.abs(xl)<z['REX'], np.abs(yl)<z['REY'])
            self.open &= isGood

        # now let's get the normal direction
        mu = np.sum(norm*self.p, axis=-1)
        mu_ = np.abs(mu)
        theta_inc = np.where(mu_<1, np.arccos(mu_), 0) # in radians

        # new direction
        n_new__n_old = n_new/self.n_loc
        self.n_loc = n_new
        # Snell's law for new direction
        p_out = self.p + ((np.sqrt(n_new__n_old**2-np.sin(theta_inc)**2) - np.cos(theta_inc))*np.where(mu>0,1,-1))[:,:,None]*norm
        pnorm = np.sum(np.abs(p_out**2),axis=-1)**.5
        p_out = p_out/pnorm[:,:,None]
        del pnorm

        # if there is an electric field
        if self.E is not None:
            if tCoefs is None:
                TS = TP = -np.ones((self.N,self.N), dtype=np.complex128)
            else:
                TS,TP = tCoefs(theta_inc)

            # S-type direction as a 3D vector
            Sdir = np.cross(norm[:,:,1:],self.p[:,:,1:])
            snorm = np.sum(np.abs(Sdir**2),axis=-1)**.5
            Sdir = Sdir/snorm[:,:,None]
            del snorm

            # P-type directions
            Pdir_in = np.cross(Sdir,self.p[:,:,1:])
            Pdir_out = np.cross(p_out[:,:,1:],Sdir)

            # E-field transformation
            tempS = TS[:,:,None] * np.sum(self.E[:,:,:,1:] * Sdir[:,:,None,:], axis=-1)
            tempP = TP[:,:,None] * np.sum(self.E[:,:,:,1:] * Pdir_in[:,:,None,:], axis=-1)
            self.E[:,:,:,1:] = tempS[:,:,:,None] * Sdir[:,:,None,:] + tempP[:,:,:,None] * Pdir_out[:,:,None,:]
            del tempS, tempP

        # update outgoing direction
        self.p=p_out

# This is based on the specifications in:
# "Opto-Mechanical Definitions", RST-SYS-SPEC-0055, Revision E
# released by the Configuration Management Office June 11, 2021
# (not export controlled)
# Most information is in CODE V format, but some was converted to be
# usable in this Python script.

def RomanRayBundle(xan, yan, N, usefilter, wl=None, hasE=False, jacobian = np.array([[1,0],[0,1]])):
    """Carries out trace through RST optics.
    xan, yan : angles in degrees in WFI local field angles
    N : pupil sampling
    usefilter : character, one of 'R', 'Z', 'Y', 'J', 'H', 'F', 'K', 'W'
    hasE : propagate electric field

    uses wavelength wl (in mm) if given; if it is None uses the central wavelength of that filter

    Returns a RayBundle object, with some added information:
    RB.x_out : shape (2,), location of output ray on FPA [in mm]
    RB.xyi : shape (N,N,2), coordinates of the initial rays [in mm]
    RB.u : shape (N,N,2), directions (orthographic projection)
    RB.s : shape (N,N), optical path length of ray to position s

    If hasE is true:
    RB.E : shape (N,N,2,4), complex, electric field for the 2 initial polarizations and 3 components (last axis 0th component should be 0)
    """

    wlref = {'R': 0.00062, 'Z': 0.00087, 'Y': 0.00106, 'J': 0.00129, 'H': 0.00158, 'F': 0.00184, 'K': 0.00213, 'W': 0.00146}[usefilter[0].upper()]
    if wl is None: wl=wlref

    # initialization
    RB = RayBundle(xan, yan, N, wl=wl, wlref=wlref, hasE=hasE, jacobian = jacobian)

    # obstructions:

    # secondary mirror support tubes
    RB.mask(build_transform_matrix(xde=-646.3906734739937, yde=-392.1337102549381, zde=2096.85,
                  ade=122.741,bde=-37.588,cde=-158.584), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])
    RB.mask(build_transform_matrix(xde=-646.3906734739937, yde=392.1337102549381, zde=2096.85,
                  ade=-122.741,bde=-37.588,cde=-21.416), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])
    RB.mask(build_transform_matrix(xde=-16.40275237234653, yde=755.8576220577872, zde=2096.85,
                  ade=-116.448,bde=-15.797,cde=-7.712), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])
    RB.mask(build_transform_matrix(xde=662.793160207391, yde=363.7239698975893, zde=2096.85,
                  ade=-155.534,bde=61.911,cde=62.718), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])
    RB.mask(build_transform_matrix(xde=662.793160207391, yde=-363.7239698975893, zde=2096.85,
                  ade=155.534,bde=61.911,cde=117.282), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])
    RB.mask(build_transform_matrix(xde=-16.40275237234653, yde=-755.8576220577872, zde=2096.85,
                  ade=116.448,bde=-15.797,cde=-172.288), 0, 0, None, [{'REX': 38.0, 'REY': 1140.0}])

    # secondary mirror baffles
    RB.mask(build_transform_matrix(zde=2892.83,ade=-180,cde=180), 0, 0, None, [{'CIR': 358., 'ADX': 15.56, 'HOL': 300.}])
    RB.mask(build_transform_matrix(zde=2892.83), 0, 0, None, [{'CIR': 358.},
            {'REX': 250., 'REY': 84.665, 'ADX': 125., 'ADY': 216.5063509, 'ARO': -120., 'iCIR_ORIG': 398.02}])
    RB.mask(build_transform_matrix(zde=2892.83), 0, 0, None, [{'CIR': 358.},
            {'REX': 250., 'REY': 84.665, 'ADX': -250., 'iCIR_ORIG': 398.02}])
    RB.mask(build_transform_matrix(zde=2892.83), 0, 0, None, [{'CIR': 358.},
            {'REX': 250., 'REY': 84.664, 'ADX': 125., 'ADY': -216.5063509, 'ARO': 120., 'iCIR_ORIG': 398.02}])

    # primary baffles
    RB.mask(build_transform_matrix(zde=697.13,ade=-180,cde=180), 0, 0, None, [{'CIR': 352.3}]) # there is a hole, but we don't account for it at this step
    RB.mask(build_transform_matrix(zde=799.518,ade=-180,cde=150), 0, 0, None, [{'CIR': 1e18, 'HOL': 1181.56}]) # flipped Boolean from original file since this is a stop

    # now the primary mirror
    RB.intersect_surface_and_reflect(build_transform_matrix(zde=660.4,ade=-180,cde=180), Rinv=-1./5671.1342, K=-0.9728630311,
        activeZone = [{'CIR': 1184.02, 'OBS': 321.31}])

    # Secondary mirror
    RB.intersect_surface_and_reflect(build_transform_matrix(zde=2945.4,ade=-180,cde=180), Rinv=-1./1299.6164, K=-1.6338521231,
        activeZone = [{'CIR': 266.255}])

    # PM hole
    RB.mask(build_transform_matrix(zde=660.4,ade=-180,cde=180), -1./5671.1342, -0.9728630311, None, [{'CIR': 1e18, 'HOL': 321.31}])

    # Fold mirror #1
    RB.intersect_surface_and_reflect(build_transform_matrix(xde=-73.371025,yde=127.0823431034063,zde=-299.6,
        ade=135.6742566218209,bde=21.96993018862709,cde=159.0416363940703), Rinv=0., K=0., activeZone = [
        {'REX': 134.13, 'REY': 152.42, 'ADY': 28.84}, {'REX': 151.11, 'REY': 135.44, 'ADY': 28.84},
        {'CIR': 16.98, 'ADX': -134.13, 'ADY': 164.28}, {'CIR': 16.98, 'ADX': 134.13, 'ADY': 164.28},
        {'CIR': 16.98, 'ADX': 134.13, 'ADY': -106.6}, {'CIR': 16.98, 'ADX': -134.13, 'ADY': -106.6}
        ])

    # Entrance aperture plate
    RB.mask(build_transform_matrix(xde=87.50199,yde=-151.5579,zde=-325.713,ade=99.2177489242795,bde=29.6785891029215,cde=175.4060606593105),
        0, 0, None, [{'iREX': 146.45, 'iREY': 94.98, 'ADY': 17.45}])

    # Fold mirror #2
    TF2 = build_transform_matrix(xde=466.3656874216886,yde=-807.7690655211503,zde=-387.2147203297053,
              ade=100.0982115641859,bde=29.61417435444774,cde=174.9705371700583)
    RB.mask(TF2, 0, 0, None, [{'CIR': 104.0, 'ADY': -197.82}])
    RB.intersect_surface_and_reflect(TF2, Rinv=0., K=0., activeZone = [
        {'CIR': 47.7, 'ADX': 169.24, 'ADY': -92.27}, {'CIR': 47.7, 'ADX': -169.24, 'ADY': -92.27},
        {'REX': 169.255, 'REY': 47.7, 'ADY': -92.27}, {'REX': 216.955, 'REY': 142.135, 'ADY': 49.865}
        ])

    # Tertiary mirror
    RB.intersect_surface_and_reflect(build_transform_matrix(xde=85.50941274300123,yde=-148.1066473962562,zde=-938.9487447474822,
        ade=117.6411883903639,bde=27.08781188751067,cde=166.5871249774839),
        Rinv=-1./1643.2784, K=-0.5965290831, activeZone = [
        {'REX': 197.435, 'REY': 207.00565, 'ADY': 222.29065}, {'REX': 302.715, 'REY': 101.72565, 'ADY': 222.29065},
        {'CIR': 105.28, 'ADX': -197.435, 'ADY': 324.0163}, {'CIR': 105.28, 'ADX':197.435, 'ADY': 324.0163},
        {'CIR': 105.28, 'ADX': 197.435, 'ADY': 120.565}, {'CIR': 105.28, 'ADX': -197.435, 'ADY': 120.565},
        {'REX': 256.189698, 'REY': 31.9859, 'ADY': 444.7891}
        ])

    # Exit pupil mask
    PupilLoc = build_transform_matrix(xde=526.6061126910644,yde=-912.1085427572655,zde=-541.0972086432624,
        ade=91.27416530783904,bde=29.99386508559931,cde=179.3629567305957)
    xy,_,_ = RB.intersect_surface(PupilLoc, update=False)
    if usefilter[0].upper() in ['F', 'K']:
        RB.mask(PupilLoc, 0, 0, 44.3, [{'CIR': 17.1, 'ADY': -0.5},
            {'REY': 1.5, 'REX': 22.2, 'ADX': -30.8563, 'ADY': 1.2268, 'ARO': 160.5},
            {'REY': 1.5, 'REX': 22.2, 'ADX': -13.9795, 'ADY': -26.6518, 'ARO': -103},
            {'REY': 1.5, 'REX': 22.2, 'ADX': 13.9795, 'ADY': -26.6518, 'ARO': -77},
            {'REY': 1.5, 'REX': 22.2, 'ADX': 30.8563, 'ADY': 1.2268, 'ARO': 19.5},
            {'REY': 1.5, 'REX': 22.2, 'ADX': 16.1804, 'ADY': 24.9799, 'ARO': 44.1},
            {'REY': 1.5, 'REX': 22.2, 'ADX': -16.1804, 'ADY': 24.9799, 'ARO': 135.9}
            ])
    else:
        RB.mask(PupilLoc, 0, 0, 47.5, [{'CIR': 12.0, 'ADY': -0.5},
            {'REY': .5, 'REX': 22.2, 'ADX': -30.8563, 'ADY': 1.2268, 'ARO': 160.5},
            {'REY': .5, 'REX': 22.2, 'ADX': -13.9795, 'ADY': -26.6518, 'ARO': -103},
            {'REY': .5, 'REX': 22.2, 'ADX': 13.9795, 'ADY': -26.6518, 'ARO': -77},
            {'REY': .5, 'REX': 22.2, 'ADX': 30.8563, 'ADY': 1.2268, 'ARO': 19.5},
            {'REY': .5, 'REX': 22.2, 'ADX': 16.1804, 'ADY': 24.9799, 'ARO': 44.1},
            {'REY': .5, 'REX': 22.2, 'ADX': -16.1804, 'ADY': 24.9799, 'ARO': 135.9}
            ])

    # Filter - Surface S1
    RB.intersect_surface_and_refract(build_transform_matrix(xde=531.5125171153529,yde=-920.606684502614,zde=-539.1713886876893,
        ade=102.76851389522,bde=29.38268469198068,cde=173.6554980927907), Rinv=-1./1500., K=0., n_new = n_Infrasil301(wlref), activeZone = [{'CIR': 52.65}])

    # Filter - Surface S2
    S2 = build_transform_matrix(xde=536.4189215396419,yde=-929.1048262479633,zde=-537.2455687321163,
        ade=102.76851389522,bde=29.38268469198068,cde=173.6554980927907)
    Rinv2 = -1./1499.31453814
    _,_,L = RB.intersect_surface(S2, Rinv=Rinv2, K=0., update=False)
    RB.s += L * (n_Infrasil301(wl)-n_Infrasil301(wlref))
    # comment - the ray trace follows the geometric path at wlref, but we include the wavelength dependence
    # in the path length.
    # This way, the DCR does not appear in the astrometry, rather it is a decentering of the PSF.
    RB.intersect_surface_and_refract(S2, Rinv=Rinv2, K=0., n_new = 1., activeZone = [{'CIR': 52.65}])

    # FPA
    TrFPA = build_transform_matrix(xde=866.9584811995454,yde=-1501.61613749036,zde=-407.5050041451383,
        ade=-62.41145131632292,bde=-27.09897706981732,cde=13.3889006733882)
    xyFPA, _, _ = RB.intersect_surface(TrFPA, Rinv=0., K=0., update=True)
    RB.u = np.einsum('ij,abj->abi', np.linalg.inv(TrFPA), RB.p)[:,:,1:3]

    # get position of central ray and update wavefront map accordingly
    RB.x_out = np.mean(xyFPA[N//2-1:N//2+1,N//2-1:N//2+1,:], axis=(0,1))
    RB.s += np.sum(RB.u*(RB.x_out[None,None,:]-xyFPA), axis=-1)

    return RB

def selftest():
    """Test functions"""

    print(build_transform_matrix(xde=-73.371025,yde=127.082343,zde=-299.600000,ade=135.674257,bde=21.969930,cde=159.041636,unit='degree'))

    RB = RayBundle(-0.071, -0.037, 2)

    print('--x--')
    print(RB.x)
    print('--p--')
    print(RB.p)

    # pupils
    RB = RomanRayBundle(-0.399, 0.208, 512, 'W', wl=9.27e-4, hasE=True)
    fits.PrimaryHDU(RB.open.astype(np.int8)).writeto('temp.fits', overwrite=True)
    fits.PrimaryHDU(np.where(RB.open,RB.s-np.median(RB.s),0)).writeto('temp-s.fits', overwrite=True)
    fits.PrimaryHDU(np.where(RB.open,RB.u[:,:,0],0)).writeto('temp-u.fits', overwrite=True)

    print('-- E out --')
    print(RB.E[128,128,:,:])
    print(RB.x[::64, ::64, 1:])
    print(RB.p[::64, ::64, 1:])
    print(RB.u[::64, ::64, :])

    print('-->', RB.x_out)

    print('-- n table --')
    for wl in np.linspace(6e-4,2.4e-3,37):
        print('{:11.5E} {:8.6f}'.format(wl, n_Infrasil301(wl)))

if __name__ == '__main__':
    selftest()
