import numpy as np
from numpy import newaxis as na

def filter_characteristic_matrix(n1,t1,n2,t2,n3,t3,ll,ux,uy):

# returns characteristic matrix of the interference filter for (vacuum) wavelength ll and angle of incidence given by sin_theta = (ux**2 + uy**2)**0.5

# Note that this function returns a pair of 2x2 matrices which are respectively the characteristic matrices for the TE and TM modes of the incident wave


    u = np.sqrt((ux**2)+(uy**2))
    print(ux,uy,u)
    k0 = 2*np.pi/ll
    kz1 = k0*np.sqrt((n1**2)-(u**2))
    kz2 = k0*np.sqrt((n2**2)-(u**2))
    kz3 = k0*np.sqrt((n3**2)-(u**2))

    e1 = n1**2
    e2 = n2**2
    e3 = n3**2

    mu1 = 1.0
    mu2 = 1.0
    mu3 = 1.0


    # Characteristic matrix of layer 1 for the TE wave 
    M_TE_1 = np.zeros((2,2),dtype=np.complex128)

    M_TE_1[0,0] = np.cos(kz1*t1)
    M_TE_1[1,1] = np.cos(kz1*t1)
    M_TE_1[0,1] = -(k0*mu1/kz1)*1j*np.sin(kz1*t1)
    M_TE_1[1,0] = -(kz1/k0*mu1)*1j*np.sin(kz1*t1)

    # Characteristic matrix of layer 1 for TM wave 
    M_TM_1 = np.zeros((2,2),dtype=np.complex128)

    M_TM_1[0,0] = np.cos(kz1*t1)
    M_TM_1[1,1] = np.cos(kz1*t1)
    M_TM_1[0,1] = -(k0*e1/kz1)*1j*np.sin(kz1*t1)
    M_TM_1[1,0] = -(kz1/k0*e1)*1j*np.sin(kz1*t1)
    
    # Characteristic matrix of layer 2 for the TE wave 
    M_TE_2 = np.zeros((2,2),dtype=np.complex128)

    M_TE_2[0,0] = np.cos(kz2*t2)
    M_TE_2[1,1] = np.cos(kz2*t2)
    M_TE_2[0,1] = -(k0*mu2/kz2)*1j*np.sin(kz2*t2)
    M_TE_2[1,0] = -(kz2/k0*mu2)*1j*np.sin(kz2*t2)

    # Characteristic matrix of layer 2 for TM wave 
    M_TM_2 = np.zeros((2,2),dtype=np.complex128)

    M_TM_2[0,0] = np.cos(kz2*t2)
    M_TM_2[1,1] = np.cos(kz2*t2)
    M_TM_2[0,1] = -(k0*e2/kz2)*1j*np.sin(kz2*t2)
    M_TM_2[1,0] = -(kz2/k0*e2)*1j*np.sin(kz2*t2)
    
    # Characteristic matrix of layer 3 for the TE wave 
    M_TE_3 = np.zeros((2,2),dtype=np.complex128)

    M_TE_3[0,0] = np.cos(kz3*t3)
    M_TE_3[1,1] = np.cos(kz3*t3)
    M_TE_3[0,1] = -(k0*mu3/kz3)*1j*np.sin(kz3*t3)
    M_TE_3[1,0] = -(kz3/k0*mu3)*1j*np.sin(kz3*t3)

    # Characteristic matrix of layer 3 for TM wave 
    M_TM_3 = np.zeros((2,2),dtype=np.complex128)

    M_TM_3[0,0] = np.cos(kz3*t3)
    M_TM_3[1,1] = np.cos(kz3*t3)
    M_TM_3[0,1] = -(k0*e3/kz3)*1j*np.sin(kz3*t3)
    M_TM_3[1,0] = -(kz3/k0*e3)*1j*np.sin(kz3*t3)
    

    
    M_TE_net = np.matmul(M_TE_1,np.matmul(M_TE_2,M_TE_3))
    M_TM_net = np.matmul(M_TM_1,np.matmul(M_TM_2,M_TM_3))

    return (M_TE_net,M_TM_net)



         
