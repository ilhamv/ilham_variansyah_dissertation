import numpy as np
from scipy.sparse import diags
import scipy as sp
import scipy.linalg
import scipy.interpolate
import scipy.optimize
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
from thermo.chemical import Chemical
import h5py

#===============================================================================
# Setup
#===============================================================================

# Thermal properties
def rho_f(T): # kg/m^3 --> kg/cc
    return Chemical('dioxouranium',T=T).rho*1E-6
def rho_w(T): # kg/m^3 --> kg/cc
    return Chemical('water',T=T,P=15.5E6).rho*1E-6
def cp_f(T): # J/kg-K
    return Chemical('dioxouranium',T=T).Cp
def cp_w(T): # J/kg-K
    return Chemical('water',T=T,P=15.5E6).Cp

# Geometry
Rf = 0.5   # cm
d  = 1.3   # cm
Z  = 360.0 # cm
Af = np.pi*Rf**2
Aw = d**2-Af

# Operational parameters
def Tw_in(t):
    if t == 0.0:
        return 565.0
    else:
        return 515.0

mdot  = 0.3   # kg/s
Pow   = 200.0 # W/cm
h     = 0.2   # W/cm^2-K

# Neutronic properties
kappa_f  = 191.4*1.6022e-13 # MeV --> J
v        = 2.2E5            # cm/s
nu      = 2.43
xi      = 0.8097
Tf       = [565.0, 1565.0, 565.0, 1565.0]
Tw       = [565.0,  565.0, 605.0,  605.0]
SigmaT   = [0.655302, 0.653976, 0.61046]
SigmaS   = [0.632765, 0.631252, 0.589171]
nuSigmaF = [xi*0.0283063, xi*0.0277754, xi*0.0265561]
SigmaT.append(SigmaT[2]+(SigmaT[1]-SigmaT[0]))
SigmaS.append(SigmaS[2]+(SigmaS[1]-SigmaS[0]))
nuSigmaF.append(nuSigmaF[2]+(nuSigmaF[1]-nuSigmaF[0]))

# Set XS interpolators
f_SigmaT   = sp.interpolate.interp2d(Tf,Tw,SigmaT)
f_SigmaS   = sp.interpolate.interp2d(Tf,Tw,SigmaS)
f_nuSigmaF = sp.interpolate.interp2d(Tf,Tw,nuSigmaF)

# Numerical parameters
tol   = 1E-5
J     = 1800
dz    = Z/J

# Spatial grid
x = np.linspace(0,Z,J+1)
x = (x[1:] + x[:-1])/2

# Simulation length
T = 1.0 # s


# =============================================================================
# Load IC
# =============================================================================

with h5py.File('ic.h5','r') as f:
    phi_ic   = f['phi_ic'][()]
    Tf_ic    = f['Tf_ic'][()]
    Tw_ic    = f['Tw_ic'][()]
    rho_ic    = f['rho_ic'][()]
    pow_ic    = f['pow_ic'][()]

#===============================================================================
# Solve TD
#===============================================================================

K_list = [20,40,60,80,100,200,400,600,800,1000]
for K in K_list:
    # Timer
    time_keff = 0.0
    time_start = time.time()
    
    # Time grid
    t_prv = 0.0
    t_nxt = 0.0
    dt    = T/K

    #===========================================================================
    # Containers
    #===========================================================================

    # XS
    SigmaA   = np.zeros(J)
    nuSigmaF = np.zeros(J)
    SigmaF   = np.zeros(J)
    D        = np.zeros(J)
    Ddz      = np.zeros(J-1)

    # TD solution
    phi_prv  = np.zeros(J)
    Tf_prv   = np.zeros(J)
    Tw_prv   = np.zeros(J)
    phi_old  = np.zeros(J)
    Tf_old   = np.zeros(J)
    Tw_old   = np.zeros(J)

    # k-eigenvalue
    phi_in   = np.ones(J)
    phi_oldi = np.ones(J)
    S        = np.ones(J)

    # For TT
    q        = 0.0
    Tin      = 0.0
    Tf_prvj  = 0.0
    Tw_prvj  = 0.0

    # Storage
    store_phi = np.zeros([K+1,J])
    store_Tf  = np.zeros([K+1,J])
    store_Tw  = np.zeros([K+1,J])
    store_rho = np.zeros(K+1)
    store_pow = np.zeros(K+1)

    #===========================================================================
    # Functions
    #===========================================================================

    # Flux normalization
    def normalize(ph):
        norm = Pow/(kappa_f*sum(np.multiply(SigmaF,ph))/J)
        ph[:] = ph*norm

    # Nonlinear problem for T
    def TT(T):
        Tf   = T[0]
        Tw   = T[1]
        r    = np.zeros_like(T)
        qs   = 2*np.pi*Rf*h*(Tf-Tw)
        r[0] = (Tf-Tf_prvj)/dt - (q-qs)/(rho_f(Tf)*Af*cp_f(Tf))
        r[1] = (Tw-Tin)/dz - (qs/cp_w(Tw) - rho_w(Tw)*Aw/dt*(Tw-Tw_prvj))/mdot
        return r
    
    #===========================================================================
    # Solve
    #===========================================================================

    # Initial condition
    phi = np.copy(phi_ic)
    Tw  = np.copy(Tw_ic)
    Tf  = np.copy(Tf_ic)
    
    # Store
    store_phi[0,:] = phi[:]
    store_rho[0]   = rho_ic
    store_pow[0]   = pow_ic
    store_Tf[0,:]  = Tf[:]
    store_Tw[0,:]  = Tw[:]

    # March in time
    for k in range(K):
        # Set previous solution
        phi_prv[:] = phi[:]
        Tf_prv[:]  = Tf[:]
        Tw_prv[:]  = Tw[:]
        t_prv = t_nxt
        t_nxt = t_prv + dt

        # Initiate piccard iteration
        err = 1.0
        while err > tol:
            # Store old solution
            phi_old[:] = phi[:]
            Tw_old[:]  = Tw[:]
            Tf_old[:]  = Tf[:]
            
            #===================================================================
            # Solve neutron diffusion
            #===================================================================

            # Set properties
            for j in range(J):
                SigmaT      = f_SigmaT(Tf[j],Tw[j])
                SigmaS      = f_SigmaS(Tf[j],Tw[j])
                nuSigmaF[j] = f_nuSigmaF(Tf[j],Tw[j])
                SigmaF[j]   = nuSigmaF[j]/nu
                SigmaA[j]   = SigmaT - SigmaS
                D[j]        = 1.0/(3*SigmaT)
            for j in range(J-1):
                Ddz[j] = 2.0/(dz/D[j]+dz/D[j+1])

            # Matrix A
            A_ul = -Ddz/dz
            A_d  = SigmaA - nuSigmaF + 1.0/(v*dt)
            for j in range(J-1):
                A_d[j] = A_d[j] - A_ul[j]
            for j in range(1,J):
                A_d[j] = A_d[j] - A_ul[j-1]
            A_d[0]   = A_d[0]   + 2*D[0]/(dz*dz)*(1.0 - 1.0/(0.25*dz/D[0]+1))
            A_d[J-1] = A_d[J-1] + 2*D[J-1]/(dz*dz)*(1.0 - 1.0/(0.25*dz/D[J-1]+1))
            A = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")

            # Preconditioner P
            Pr = spla.spilu(A)
            Pr = spla.LinearOperator(A.shape, Pr.solve)

            # Solve neutron
            phi[:], exitCode = gmres(A, phi_prv/(v*dt), x0=phi , M=Pr, tol=tol*1E-2)

            # Error neutron
            phi_old[:] = phi - phi_old # phi_old stores error
            err_phi    = np.linalg.norm(abs(np.true_divide(phi_old,phi)))
                
            #===================================================================
            # Solve heat transfer
            #===================================================================

            # Solve T
            Tin = Tw_in(t_nxt)
            for j in range(J):
                # Solve Tw
                q = kappa_f*SigmaF[j]*phi[j]
                Tf_prvj = Tf_prv[j]
                Tw_prvj = Tw_prv[j]
                [Tf[j], Tw[j]] = sp.optimize.fsolve(TT,[Tf[j],Tw[j]],xtol=tol*1E-2)              
                # Reset Tin
                Tin = Tw[j]

            # Error T and all
            Tw_old[:] = Tw - Tw_old
            err_Tw    = np.linalg.norm(abs(np.true_divide(Tw_old,Tw)))
            Tf_old[:] = Tf - Tf_old
            err_Tf    = np.linalg.norm(abs(np.true_divide(Tf_old,Tf)))
            errT      = max(err_Tw,err_Tf)
            err       = max(errT,err_phi)
            print(K,k,err_phi,err_Tf,err_Tw)

        # Store solution
        store_phi[k+1,:] = phi[:]
        store_Tf[k+1,:]  = Tf[:]
        store_Tw[k+1,:]  = Tw[:]
        store_pow[k+1]   = kappa_f*sum(np.multiply(SigmaF,phi))/J

        #=======================================================================
        # Solve k-eigenvalue to get rho
        #=======================================================================

        time_keff_start = time.time()
        
        # Matrix A
        A_ul = -Ddz/dz
        A_d  = SigmaA
        for j in range(J-1):
            A_d[j] = A_d[j] - A_ul[j]
        for j in range(1,J):
            A_d[j] = A_d[j] - A_ul[j-1]
        A_d[0]   = A_d[0]   + 2*D[0]/(dz*dz)*(1.0 - 1.0/(0.25*dz/D[0]+1))
        A_d[J-1] = A_d[J-1] + 2*D[J-1]/(dz*dz)*(1.0 - 1.0/(0.25*dz/D[J-1]+1))
        A = diags([A_ul,A_d,A_ul],[-1,0,1],format="csc")

        # Preconditioner P
        Pr = spla.spilu(A)
        Pr = spla.LinearOperator(A.shape, Pr.solve)

        # Prepare eigenvalue solve
        erri      = 1.0
        phi_in[:] = phi[:]
        keff      = 1.0
        while erri > tol:
            # Store old solution
            phi_oldi[:] = phi_in[:]
            k_oldi = keff

            # Set fission source
            S[:] = np.multiply(nuSigmaF,phi_in)/keff

            # Solve
            phi_in[:], exitCode = gmres(A, S, x0=phi_in, M=Pr, tol=tol*1E-2)
            keff = k_oldi*sum(np.multiply(nuSigmaF,phi_in))/sum(np.multiply(nuSigmaF,phi_oldi))

            # Normalize
            normalize(phi_in)

            # Error neutron
            phi_oldi[:] = phi_in - phi_oldi # phi_oldi stores error
            err_phi    = np.linalg.norm(abs(np.true_divide(phi_oldi,phi_in)))
            err_k      = abs((keff-k_oldi)/keff)
            erri = max(err_k,err_phi)
        store_rho[k+1] = (keff-1)/keff
        print(K,k,"keff",keff,(keff-1)/keff)
        
        # Monitor
        '''
        plt.plot(store_rho)
        plt.show()
        
        plt.plot(store_pow)
        plt.show()
        
        plt.plot(phi)
        plt.show()
        
        plt.plot(Tf)
        plt.show()
        
        plt.plot(Tw)
        plt.show()
        '''
        
        time_keff_end  = time.time()
        time_keff     += time_keff_end - time_keff_start
    time_end     = time.time()
    time_elapsed = time_end - time_start - time_keff
    print("time",time_elapsed,time_keff)
    with h5py.File("td_%i.h5"%K, 'w') as hdf:
        hdf.create_dataset('t',   data=np.linspace(0,T,K+1))
        hdf.create_dataset('te',  data=time_elapsed)
        hdf.create_dataset('tek', data=time_keff)
        hdf.create_dataset('phi', data=store_phi)
        hdf.create_dataset('Tf',  data=store_Tf)
        hdf.create_dataset('Tw',  data=store_Tw)
        hdf.create_dataset('rho', data=store_rho)
        hdf.create_dataset('pow', data=store_pow)
