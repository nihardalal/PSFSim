import numpy as np
from numpy import newaxis as na
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
class filter_detector(object):
    
    def __init__(self,n1,t1,n2,t2,n3,t3,sgn):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.e1 = n1**2
        self.e2 = n2**2
        self.e3 = n3**2

        self.mu1 = 1.0
        self.mu2 = 1.0
        self.mu3 = 1.0
        self.muHgCdTe = 1.0
        
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.sgn = sgn


    def characteristic_matrix(self,ll,ux,uy):
        print('Calculating characteristic matrices......')
        start_time = time.time() 
    #   returns characteristic matrix of the interference filter for (vacuum) wavelength ll and     angle of incidence given by sin_theta = (ux**2 + uy**2)**0.5

    #    Note that this function returns a pair of 2x2 matrices which are respectively the chara    cteristic matrices for the TE and TM modes of the incident wave
        try:
            shape = ux.shape
        except:
            shape=(1,1)
        mask = (ux**2 + uy**2) <= 1.0
        #mask = np.abs(ux) + np.abs(uy) <= 1.0
        u = np.sqrt((ux**2)+(uy**2))

        
        k0 = 2*np.pi/ll
        kz1 = np.zeros_like(ux,dtype=np.complex128)
        kz1[mask] = k0*np.sqrt((self.n1**2)-(u[mask]**2))
        kz2 = np.zeros_like(ux,dtype=np.complex128)
        kz2[mask] = k0*np.sqrt((self.n2**2)-(u[mask]**2))
        kz3 = np.zeros_like(ux,dtype=np.complex128)
        kz3[mask] = k0*np.sqrt((self.n3**2)-(u[mask]**2))
       

        self.e1 = self.n1**2
        self.e2 = self.n2**2
        self.e3 = self.n3**2

        self.mu1 = 1.0
        self.mu2 = 1.0
        self.mu3 = 1.0


        # Characteristic matrix of layer 1 for the TE wave 
        M_TE_1 = np.zeros(shape+(2,2),dtype=np.complex128)

        M_TE_1[mask,0,0] = np.cos(kz1[mask]*self.t1)
        M_TE_1[mask, 1,1] = np.cos(kz1[mask]*self.t1)
        M_TE_1[mask,0,1] = -(k0*self.mu1/kz1[mask])*1j*np.sin(kz1[mask]*self.t1)
        M_TE_1[mask,1,0] = -(kz1[mask]/k0/self.mu1)*1j*np.sin(kz1[mask]*self.t1)

        # Characteristic matrix of layer 1 for TM wave 
        M_TM_1 = np.zeros(shape+(2,2),dtype=np.complex128)
   
        M_TM_1[mask,0,0] = np.cos(kz1[mask]*self.t1)
        M_TM_1[mask,1,1] = np.cos(kz1[mask]*self.t1)
        M_TM_1[mask,0,1] = -(k0*self.e1/kz1[mask])*1j*np.sin(kz1[mask]*self.t1)
        M_TM_1[mask,1,0] = -(kz1[mask]/k0/self.e1)*1j*np.sin(kz1[mask]*self.t1)
        
        # Characteristic matrix of layer 2 for the TE wave 
        M_TE_2 = np.zeros(shape+(2,2),dtype=np.complex128)

        M_TE_2[mask,0,0] = np.cos(kz2[mask]*self.t2)
        M_TE_2[mask,1,1] = np.cos(kz2[mask]*self.t2)
        M_TE_2[mask,0,1] = -(k0*self.mu2/kz2[mask])*1j*np.sin(kz2[mask]*self.t2)
        M_TE_2[mask,1,0] = -(kz2[mask]/k0/self.mu2)*1j*np.sin(kz2[mask]*self.t2)

        # Characteristic matrix of layer 2 for TM wave 
        M_TM_2 = np.zeros(shape+(2,2),dtype=np.complex128)

        M_TM_2[mask,0,0] = np.cos(kz2[mask]*self.t2)
        M_TM_2[mask,1,1] = np.cos(kz2[mask]*self.t2)
        M_TM_2[mask,0,1] = -(k0*self.e2/kz2[mask])*1j*np.sin(kz2[mask]*self.t2)
        M_TM_2[mask,1,0] = -(kz2[mask]/k0/self.e2)*1j*np.sin(kz2[mask]*self.t2)
        
        # Characteristic matrix of layer 3 for the TE wave 
        M_TE_3 = np.zeros(shape+(2,2),dtype=np.complex128)

        M_TE_3[mask,0,0] = np.cos(kz3[mask]*self.t3)
        M_TE_3[mask,1,1] = np.cos(kz3[mask]*self.t3)
        M_TE_3[mask,0,1] = -(k0*self.mu3/kz3[mask])*1j*np.sin(kz3[mask]*self.t3)
        M_TE_3[mask,1,0] = -(kz3[mask]/k0/self.mu3)*1j*np.sin(kz3[mask]*self.t3)

        # Characteristic matrix of layer 3 for TM wave 
        M_TM_3 = np.zeros(shape+(2,2),dtype=np.complex128)

        M_TM_3[mask,0,0] = np.cos(kz3[mask]*self.t3)
        M_TM_3[mask,1,1] = np.cos(kz3[mask]*self.t3)
        M_TM_3[mask,0,1] = -(k0*self.e3/kz3[mask])*1j*np.sin(kz3[mask]*self.t3)
        M_TM_3[mask,1,0] = -(kz3[mask]/k0/self.e3)*1j*np.sin(kz3[mask]*self.t3)
        

        
        M_TE_net = np.matmul(M_TE_1,np.matmul(M_TE_2,M_TE_3))
        M_TM_net = np.matmul(M_TM_1,np.matmul(M_TM_2,M_TM_3))

        end_time = time.time()
        print(f'Finished computing characteristic matrices in {end_time-start_time:.3f}')

        return {'TE':M_TE_net, 'TM':M_TM_net}


    def local_to_FPA_rotation(self, ux, uy):
        print('Computing local to FPA rotation.......')
        start_time = time.time()
        u = np.sqrt((ux**2)+(uy**2))
        mask = (u <= 1)
        #mask = np.abs(ux) + np.abs(uy) <= 1
        try: 
            shape = ux.shape
        except:
            shape=(1,1)
        RT = np.zeros(shape+(3,3),dtype=np.float64)
        
        RT[mask & (u == 0)] = np.identity(3)
        RT[mask & (u != 0),0,0] = (uy[mask & (u != 0)]*self.sgn/u[mask & (u != 0)])
        RT[mask & (u != 0),0,1] = (ux[mask & (u != 0)]/u[mask & (u != 0)])
        RT[mask & (u != 0), 1,0] = (-(ux[mask & (u != 0)]*self.sgn/u[mask & (u != 0)]))
        RT[mask & (u != 0),1,1] = (uy[mask & (u != 0)]/u[mask & (u != 0)])
        RT[mask & (u != 0),2,2] = self.sgn

        end_time = time.time()
        print(f'Finished computing local to FPA rotation in {end_time-start_time:.3f}')
        
        return RT


    def polarisation_mode_decomposition(self, ux, uy, Ex, Ey, Ez):

    # Function to decompose incident electric field (specified by components Ex, Ey, Ez along FPA axes) into TE and TM modes 

        u = np.sqrt((ux**2)+(uy**2))
        try:
            shape = ux.shape
        except:
            shape=(1,1)
        mask = (u <= 1)
        #mask = np.abs(ux) + np.abs(uy) <= 1

        A_TE = np.zeros(shape, dtype=np.complex128)
        A_TM = np.zeros(shape, dtype=np.complex128)

        A_TE[mask & (u == 0)] = Ex[mask & (u == 0)]
        A_TM[mask & (u == 0)] = -Ey[mask & (u == 0)]

        ek1 = np.zeros_like(ux,dtype=np.complex128)
        ek2 = np.zeros_like(ux,dtype=np.complex128)
        ek3 = np.zeros_like(ux,dtype=np.complex128)

        ek1[mask & (u != 0)] = -(ux[mask & (u != 0)]/u[mask & (u != 0)])*np.sqrt(1-(u[mask & (u != 0)]**2))
        ek2[mask & (u != 0)] = -(uy[mask & (u != 0)]/u[mask & (u != 0)])*np.sqrt(1-(u[mask & (u != 0)]**2))
        ek3[mask & (u != 0)] = u[mask & (u != 0)]*self.sgn

        A_TE[mask & (u != 0)] = (Ex[mask & (u != 0)]*(uy[mask & (u != 0)]/u[mask & (u != 0)]))-(Ey[mask & (u != 0)]*(ux[mask & (u != 0)]/u[mask & (u != 0)]))
        A_TM[mask & (u != 0)] = (ek1[mask & (u != 0)]*Ex[mask & (u != 0)])+(ek2[mask & (u != 0)]*Ey[mask & (u != 0)])+(ek3[mask & (u != 0)]*Ez[mask & (u != 0)])
        
        return {'TE':A_TE, 'TM':A_TM}   
 
    def unpolarised_mode_decomposition(self, ux, uy, E0=1.e10):

        print('Computing polarisation mode decomposition for unpolarised incident E field.....')
        start_time = time.time()
    
    # Function to obtain TE and TM mode amplitudes for unpolarised incident wave with magnitude of electric field E0.
        u = np.sqrt((ux**2)+(uy**2))
        mask = (u <= 1)
        #mask = np.abs(ux) + np.abs(uy) <= 1.0
        try:
            shape = ux.shape
        except:
            shape=(1,1)
        #shape = ux.shape
        A_TE = np.zeros(shape, dtype=np.complex128)
        A_TM = np.zeros(shape, dtype=np.complex128)

        A_TE[mask] = (1./np.sqrt(2))*E0
        A_TM[mask] = -(1./np.sqrt(2))*E0
        
        end_time = time.time()
        print(f'Finished computing polarisation mode decomposition in {end_time-start_time:.3f}')
        return {'TE':A_TE, 'TM':A_TM}   


        

    def Transmission(self,ll, ux, uy, use_HgCdTe=True):

        """
        Returns Transmission coefficients of TE and TM modes coming with incident direction (ux, uy, sqrt(1-u^2))
        """
        print('Computing transmission of E field through the interference filter....')
        start_time = time.time()
        u = np.sqrt((ux**2)+(uy**2))
        mask = (u <= 1)
        #mask = np.abs(ux) + np.abs(uy) <= 1.0
        try:
            shape = ux.shape
        except:
            shape=(1,1)

        # Characteristic matrices for the TE and TM modes
        char_matrices = self.characteristic_matrix(ll,ux,uy)
        M_TE = char_matrices['TE']
        M_TM = char_matrices['TM']
        
        if use_HgCdTe:
            nHgCdTe = self.nHgCdTe(ll)
            eHgCdTe = nHgCdTe**2
        else:
            nHgCdTe = 1.0
            eHgCdTe = nHgCdTe**2
        
        # cosine of the angle of incidence
        cos_theta = np.zeros_like(ux,dtype=np.complex128)
        cos_theta[mask] = np.sqrt(1-(ux[mask]**2)-(uy[mask]**2))
        

        k0 = 2*np.pi/ll
        kz = np.zeros_like(ux,dtype=np.complex128)
        kz[mask] = np.sqrt((k0**2)*((nHgCdTe**2)-(u[mask]**2)))
        
        kz[mask & (kz.imag < 0.)] = -kz[mask & (kz.imag < 0.)] # choose the root with positive imaginary part

        Transmission_TE = np.zeros_like(ux,dtype=np.complex128)
        Transmission_TM = np.zeros_like(ux,dtype=np.complex128)


        Transmission_TE[mask] = 2*cos_theta[mask]/((cos_theta[mask]*(M_TE[mask,0,0]+((kz[mask]/k0/self.muHgCdTe)*M_TE[mask,0,1])))+(M_TE[mask,1,0]+((kz[mask]/k0/self.muHgCdTe)*M_TE[mask,1,1])))

        Transmission_TM[mask] = 2*cos_theta[mask]/((cos_theta[mask]*(M_TM[mask,0,0]+((kz[mask]/k0/eHgCdTe)*M_TM[mask,0,1])))+(M_TM[mask,1,0]+((kz[mask]/k0/eHgCdTe)*M_TM[mask,1,1])))
        end_time = time.time()
        print(f'Computed transmission in {end_time-start_time:.3f}')
        return {'TE':Transmission_TE, 'TM':Transmission_TM}

    def Transmitted_E(self, ll, ux, uy, zp, A_TE=1.e10 ,A_TM=1.e10, use_nHgCdTe=True):        
 
        ''' Returns the transmitted Electric field components in terms of the incident amplitudes in the TE and TM polarisation modes at positions xp, yp, zp w.r.t the point of incidence (i.e. the pixel centre on the SCA).

     (xp, yp, zp) are coordinates along the FPA coordinate axes but with origin shifted to the point of incidence. The returned electric field does not include dependence on xp and yp (see notes)
        '''
        start_time = time.time()
        current_time = start_time
        u = np.sqrt((ux**2)+(uy**2))
        mask = (u <= 1)
        #mask = np.abs(ux) + np.abs(uy) <= 1.0
        shape = ux.shape

        if use_nHgCdTe:
            nHgCdTe = self.nHgCdTe(ll)
            eHgCdTe = nHgCdTe**2
        else:
            nHgCdTe = 1.0
            eHgCdTe = nHgCdTe**2
        
        k0 = 2*np.pi/ll

        kz = np.zeros_like(ux,dtype=np.complex128)
        kz[mask] = np.sqrt((k0**2)*((nHgCdTe**2)-(u[mask]**2)))
        kz[mask & (kz.imag < 0.)] = -kz[mask & (kz.imag < 0.)] # choose the root with positive imaginary part
        
        T_coeff = self.Transmission(ll, ux, uy)
        Transmission_TE = T_coeff['TE']
        Transmission_TM = T_coeff['TM']

        E_local = np.zeros(shape+(3,),dtype=np.complex128)


        E_local_x = A_TE*Transmission_TE
        H_local_x = A_TM*Transmission_TM

        E_local[mask, 0] = E_local_x[mask]
        E_local[mask, 1] = (1j/(k0*eHgCdTe))*(1j*kz[mask])*H_local_x[mask]
        E_local[mask, 2] = (u[mask]/eHgCdTe)*H_local_x[mask]

       # E_local[mask, 0] *= np.exp((1j*kz[mask]*self.sgn*zp[mask]))
       # E_local[mask, 1] *= np.exp((1j*kz[mask]*self.sgn*zp[mask]))
       # E_local[mask, 2] *= np.exp((1j*kz[mask]*self.sgn*zp[mask]))
        # Resolve components of the E-field along local x and y axes into components along FPA x and y axes

        local_to_FPA = self.local_to_FPA_rotation(ux,uy)

        E_FPA_x = np.zeros_like(ux,dtype=np.complex128)
        E_FPA_y = np.zeros_like(ux,dtype=np.complex128)
        E_FPA_z = np.zeros_like(ux,dtype=np.complex128)

        E_FPA_x[mask] = np.sum(local_to_FPA[mask, 0,:]*E_local[mask, :], axis =-1)
        E_FPA_y[mask] = np.sum(local_to_FPA[mask, 1,:]*E_local[mask, :], axis =-1)
        E_FPA_z[mask] = np.sum(local_to_FPA[mask, 2,:]*E_local[mask, :], axis =-1)

        print('Calculation of E_FPA done in ', time.time() - current_time, ' seconds')
        current_time = time.time()
        phase = np.zeros(ux.shape+zp.shape,dtype=np.complex128)
        ##log_phase = np.zeros(ux.shape+zp.shape,dtype=np.complex128)
        ##np.multiply(1j*kz[mask, na], self.sgn*zp[na, :], out=log_phase[mask, :])
        ##np.exp(log_phase, out=phase)
        phase[:,:, :] = np.exp((kz[:,:, na]*(1j*self.sgn*zp[na,na,:])))

        print('Calculation of exp done in ', time.time() - current_time, ' seconds')
        current_time = time.time()

        Ex = np.empty(ux.shape + zp.shape,dtype=np.complex128)
        Ey = np.empty(ux.shape + zp.shape,dtype=np.complex128)
        Ez = np.empty(ux.shape + zp.shape,dtype=np.complex128)

        np.multiply(E_FPA_x[:,:,na], phase, out=Ex)
        np.multiply(E_FPA_y[:,:,na], phase, out=Ey)
        np.multiply(E_FPA_z[:,:,na], phase, out=Ez)

        end_time = time.time()
        print('Calculation of phase done in ', end_time - current_time, ' seconds')
        print('Total calculation done in ', end_time - start_time, ' seconds')

        return (Ex, Ey, Ez)

    



        

         
    def nHgCdTe(self,wavelength, force_old=False, force_short=False):

    # Computes the (complex) refractive index for HgCdTe for a given wavelength
    # Can probably make this a helper function at a later stage 

        E = 1.23984198405504/wavelength # in eV
        T = 89.
        x = 0.445

        # Djuri\v si\'c & Li, J. Appl. Phys.85:2854 (1999)

        # parameters: Table II: a0i*(1-x) + (a1i+a2i*x)*x*(1-x) + a3i*x
        def f(a0,a1,a2,a3,x):
            return(a0*(1-x)+(a1+a2*x)*x*(1-x)+a3*x)

        epsilon_inf = f(1.665,0.644,-1.339,1.252,x)
        epsilon_inf = 1.938

        # E0 transition
        E0 = -0.302 +1.93*x + 5.35*(1-2*x)*T/1e4 - 0.81*x**2 + 0.832*x**3
        Delta0 = f(0.654,0.778,-2.626,2.456,x) - E0
        Gamma0 = f(0.025,6.119,-1.784,0.487,x)/10.
        A = f(3.100,-6.988,1.974,5.098,x)
        Delta0 = 1.544-E0; Gamma0 = 0.0358; A = 1.003
        chi0 = (E+1j*Gamma0)/E0
        chi0s = (E+1j*Gamma0)/(E0+Delta0)
        epsilonI = A/E0**1.5*(  1/chi0**2*(2-np.sqrt(1+chi0)-np.sqrt(1-chi0))
          + 0.5*(E0/(E0+Delta0))**1.5/chi0s**2*(2-np.sqrt(1+chi0s)-np.sqrt(1-chi0s)) )
        #
        # exciton contribution
        epsilon0x = 0.
        A0x = f(0.003,2.505,-3.000,0.010,x)
        G03D = f(0.002,3.084,-3.589,0.001,x)
        A0x = 0.011; G03D = 0.013
        for m in range(1,100):
            epsilon0x += A0x/m**3/(E0 - G03D/m**2 - E - 1j*Gamma0)

        # E1 transition
        E1 = 2.147+0.44*x+0.7*x**2
        Delta1 = 2.778+0.47*x+0.6*x**2 - E1
        Gamma1 = f(1.349,3.421,10.095,1.904,x)/10.
        B1 = f(1.320,5.542,-3.599,1.816,x)
        B1s = f(0.240,4.289,-4.040,1.055,x)
        Gamma1 = 0.2307; B1 = 1.726; B1s = 0.156
        chi1 = (E+1j*Gamma1)/E1
        chi1s = (E+1j*Gamma1)/(E1+Delta1)
        epsilonII = -B1/chi1**2*np.log(1-chi1**2) - B1s/chi1s**2*np.log(1-chi1s**2)
        #
        # exciton contribution
        epsilonIII = 0.
        B1x = f(0.453,2.604,-0.066,1.029,x)
        B2x = f(0.248,1.100,0.373,0.666,x)
        G1 = f(0.054,-0.378,0.664,0.001,x)
        G1s = f(0.057,-0.212,0.384,0.013,x)
        B1x = 0.746; B2x = 0.356; G1 = 1e-4; G1s = 1e-3
        for m in range(1,100):
            nn = 2*m-1
            epsilonIII += 1/nn**3*( B1x/(E1-G1/nn**2-E-1j*Gamma1) + B2x/(E1+Delta1-G1s/nn**2-E-1j*Gamma1) )

        # higher critical points
        #
        epsilonIV = 0.
        # sum fj^2 / (Ej^2 - E^2 - i E Gammaj')
        f2 = f(4.526,6.421,-9.926,5.232,x)
        Gamma2 = f(1.167,0.873,0.481,0.734,x)
        alpha2 = f(0.397,-0.010,0.119,0.022,x)
        E2 = f(4.559,0.506,-0.856,5.089,x)
        f2 = 4.407; Gamma2 = 1.082; alpha2 = 0.029; E2 = 4.798
        Gamma2p = Gamma2*np.exp(-alpha2*((E-E2)/Gamma2)**2)
        epsilonIV += f2**2/(E2**2-E**2-1j*E*Gamma2p)
        #
        f3 = f(6.147,6.139,-8.472,1.861,x)
        Gamma3 = f(1.946,0.808,-0.944,0.087,x)
        alpha3 = f(0.173,-1.905,4.202,0.160,x)
        E3 = f(6.499,1.564,1.374,6.356,x)
        f3 = 4.422; Gamma3 = 3.726; alpha3 = 0.081; E3 = 5.432
        Gamma3p = Gamma3*np.exp(-alpha3*((E-E3)/Gamma3)**2)
        epsilonIV += f3**2/(E3**2-E**2-1j*E*Gamma3p)
        #
        f4 = f(8.750,0,0,2.033,x)
        Gamma4 = f(4.146,0,0,0.929,x)
        alpha4 = f(0.797,0,0,0.014,x)
        E4 = f(2.919,3.472,-6.331,2.890,x)
        f4 = 8.122; Gamma4 = 4.854; alpha4 = 0.066; E4 = 3.905
        Gamma4p = Gamma4*np.exp(-alpha4*((E-E4)/Gamma4)**2)
        epsilonIV += f4**2/(E4**2-E**2-1j*E*Gamma4p)

        n = np.sqrt(epsilon_inf + epsilonI + epsilon0x + epsilonII + epsilonIII + epsilonIV)

        # conversion to MCT221
        if ((wavelength>1.000001) or force_old) and (not force_short):
            alphag = -65+1.88*T+(8694-10.31*T)*x
            Eg = -0.295+1.87*x-0.28*x**2+(6-14*x+3*x**2)/1e4*T+.35*x**4
            E = 1.23984198405504/wavelength
            beta = -1+.083*T + (21-.13*T)*x
            if E>Eg:
                alpha = alphag*np.exp(np.sqrt(beta*(E-Eg))) / 1e4 # per micron
            else:
                alpha0 = np.exp(-18.5+45.68*x)
                E0 = -0.355+1.77*x
                alpha = alpha0 * np.exp((E-E0)/(Eg-E0)*np.log(alphag/alpha0)) / 1e4 # per micron
            n_imag = alpha*wavelength/4./np.pi

            new_wt = 1.
            if not force_old:
                if wavelength>1.500001:
                    n += (n_imag-np.imag(n))*new_wt*1j
                else:
                    n_imag2 = 0.626123540873-0.100178870601*wavelength-0.165418727091*wavelength**2+0.051115994240*wavelength**3
                    n += (n_imag2-np.imag(n))*new_wt*1j
            if force_old: n+= (n_imag-np.imag(n))*1j
        

        
        return(n)

    

