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
Tw_in = 565.0 # K
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
T = 20.0 # s

# Containers
SigmaA   = np.zeros(J)
nuSigmaF = np.zeros(J)
SigmaF   = np.zeros(J)
D        = np.zeros(J)
Ddz      = np.zeros(J-1)
S        = np.zeros(J)
q        = 0.0
Tin      = 0.0


#===============================================================================
# Get the eigenvectors that correspond to the near critical system with the 
# given xi
#===============================================================================

omega = 0.5 # Piccard relaxation parameter

# Normalize flux
def normalize(phi):
    norm = Pow/(kappa_f*sum(np.multiply(SigmaF,phi))/J)
    phi[:] = phi*norm

# Temperature non-linear problem
def TT(T):
    return (T-Tin)/dz - q/(mdot*cp_w(T))

# First guess
phi = np.ones(J)
phi_old = np.ones(J)
phi_oldi = np.ones(J) # For inner flux solve
Tw_old = np.ones(J)
Tf_old = np.ones(J)
Tf  = np.ones(J)*565
Tw  = np.ones(J)*565
k   = 1.0
err = 1.0

while err > tol:
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

    # Store old solution
    phi_old[:] = phi[:]
    Tw_old[:] = Tw[:]
    Tf_old[:] = Tf[:]
    k_old = k
    
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

    # Eigenvalue flux solve
    erri = 1.0
    while erri > tol:
        # Store old solution
        phi_oldi[:] = phi[:]
        k_oldi = k

        # Set fission source
        S[:] = np.multiply(nuSigmaF,phi)/k

        # Solve
        phi[:], exitCode = gmres(A, S, x0=phi , M=Pr, tol=tol*1E-2)
        k = k_oldi*sum(np.multiply(nuSigmaF,phi))/sum(np.multiply(nuSigmaF,phi_oldi))

        # Normalize
        normalize(phi)

        # Error neutron
        phi_oldi[:] = phi - phi_oldi # phi_oldi stores error
        err_phi    = np.linalg.norm(abs(np.true_divide(phi_oldi,phi)))
        err_k      = abs((k-k_oldi)/k)
        erri = max(err_k,err_phi)
        
    # Error neutron
    phi_old[:] = phi - phi_old # phi_old stores error
    err_phi    = np.linalg.norm(abs(np.true_divide(phi_old,phi)))
    err_k      = abs((k-k_old)/k)
    err = max(err_k,err_phi)

    # Solve T
    Tin = Tw_in
    for j in range(J):
        # Solve Tw
        q = kappa_f*SigmaF[j]*phi[j]
        Tw[j] = sp.optimize.fsolve(TT,Tw[j],xtol=tol*1E-2)

        # Solve Tf
        Tf[j] = Tw[j] + q/(2*np.pi*Rf*h)

        # Get Tout
        Tin = Tw[j]

    Tw[:] = omega*Tw + (1.0-omega)*Tw_old
    Tf[:] = omega*Tf + (1.0-omega)*Tf_old

    # Error T and all
    Tw_old[:] = Tw - Tw_old
    err_Tw    = np.linalg.norm(abs(np.true_divide(Tw_old,Tw)))
    Tf_old[:] = Tf - Tf_old
    err_Tf    = np.linalg.norm(abs(np.true_divide(Tf_old,Tf)))
    errT      = max(err_Tw,err_Tf)
    err       = max(err,errT)

    print("ic",err_phi,err_k,err_Tw,err_Tf,err,k)
print(k)

# Set as initial condition for time dependent problem
for j in range(J):
    SigmaT      = f_SigmaT(Tf[j],Tw[j])
    SigmaS      = f_SigmaS(Tf[j],Tw[j])
    nuSigmaF[j] = f_nuSigmaF(Tf[j],Tw[j])
    SigmaF[j]   = nuSigmaF[j]/nu
    SigmaA[j]   = SigmaT - SigmaS
    D[j]        = 1.0/(3*SigmaT)
for j in range(J-1):
    Ddz[j] = 2.0/(dz/D[j]+dz/D[j+1])
normalize(phi)
phi_ic = np.copy(phi)
Tf_ic = np.copy(Tf)
Tw_ic = np.copy(Tw)
pow_ic = kappa_f*sum(np.multiply(SigmaF,phi))/J
k_ic = k
rho_ic = (k_ic-1.0)/k_ic

print("ic done",k_ic,rho_ic,pow_ic)

# =============================================================================
# Save initial condition
# =============================================================================

with h5py.File("ic.h5", 'w') as hdf:
    hdf.create_dataset('phi_ic', data=phi_ic)
    hdf.create_dataset('Tf_ic',  data=Tf_ic)
    hdf.create_dataset('Tw_ic',  data=Tw_ic)
    hdf.create_dataset('rho_ic',  data=rho_ic)
    hdf.create_dataset('pow_ic',  data=pow_ic)

# =============================================================================
# Plot IC
# =============================================================================

plt.plot(x,phi_ic,'k')
plt.grid()
plt.xlabel('Axial length (cm)')
plt.ylabel('Neutron Flux (/s)')
plt.savefig('phi_ic.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

plt.plot(x,Tw_ic,'b')
plt.grid()
plt.xlabel('Axial length (cm)')
plt.ylabel('Water temperature (K)')
plt.savefig('Tw_ic.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

plt.plot(x,Tf_ic,'r')
plt.grid()
plt.xlabel('Axial length (cm)')
plt.ylabel('Fuel temperature (K)')
plt.savefig('Tf_ic.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()