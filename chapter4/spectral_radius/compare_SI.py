import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import integrate
import solvers

i = complex(0,1)

#===============================================================================
# BE
#===============================================================================

def rho_BE(eta,c,fac=1.0):
    return c/(1.0 + fac/eta)

def BE_omega(lam,eta,c,fac=1.0):
    if lam == 0.0:
        return rho_BE(eta,c,fac)
    return c/lam * np.arctan(lam/(1.0+fac/eta))

def BE_omega_DSA(lam,eta,c,fac=1.0):
    omega_SI = BE_omega(lam,eta,c,fac)
    return omega_SI - c*(1.0-omega_SI)/(1.0-c+fac/eta+lam**2/(3.0*(1.0+fac/eta)))

def BE_optimize_DSA(lam,eta,c,fac=1.0):
    return -BE_omega_DSA(lam,eta,c,fac)

#===============================================================================
# MBTD
#===============================================================================

def rho_MBTD(eta,c):
    return c/(1.0 + 2.0/eta + 2.0/eta**2)**0.5

def MB_f1(mu,lam):
    return (1.0 - i*lam*mu) / (1.0 + (lam*mu)**2)

def MB_f2(mu,lam,eta):
    return (1.0 + 2.0/eta - i*lam*mu) / ((1.0 + 2.0/eta)**2 + (lam*mu)**2)

def MB_f3(mu,lam,eta):
    return 1.0 + 2.0/eta**2 * MB_f1(mu,lam) * MB_f2(mu,lam,eta)

def MB_IG1(mu,lam,eta):
    return (MB_f1(mu,lam) / MB_f3(mu,lam,eta)).real

def MB_IG2(mu,lam,eta):
    return (MB_f2(mu,lam,eta) / MB_f3(mu,lam,eta)).real

def MB_IG12(mu,lam,eta):
    return (MB_f1(mu,lam)*MB_f2(mu,lam,eta) / MB_f3(mu,lam,eta)).real

def MB_omega(lam,eta,c):
    I1  = integrate.quad(MB_IG1, -1,1,args=(lam,eta))[0]
    I2  = integrate.quad(MB_IG2, -1,1,args=(lam,eta))[0]
    I12 = integrate.quad(MB_IG12,-1,1,args=(lam,eta))[0]

    a_hat = 1.0
    b_hat = -c/2.0 * (I1+I2)
    c_hat = (c/2)**2*I1*I2 + 1.0/2.0*(c/eta*I12)**2

    D = b_hat**2 - 4.0*a_hat*c_hat

    if D<0:
        o1 = (-b_hat + i*(-D)**0.5)/(2.0*a_hat)
        o2 = (-b_hat - i*(-D)**0.5)/(2.0*a_hat)
    else:
        o1 = (-b_hat + D**0.5)/(2.0*a_hat)
        o2 = (-b_hat - D**0.5)/(2.0*a_hat)

    return o1,o2

def MB_optimize(lam,eta,c):
    o1,o2 = MB_omega(lam,eta,c)
    o1 = abs(o1)
    o2 = abs(o2)
    return -max(o1,o2)

def MB_C1hat(lam,eta):
    return lam**2/(3.0*(1.0+1.0/eta*2.0/(eta+2.0)))

def MB_C2hat(lam,eta):
    return lam**2/(3*(1.0+2.0/eta*(1.0+1.0/eta)))

def MB_C1(lam,eta,c):
    return MB_C1hat(lam,eta) + 1.0 - c

def MB_C2(lam,eta):
    return -1.0/(eta+2.0) * MB_C1hat(lam,eta)+1.0/eta

def MB_C3(lam,eta,c):
    return MB_C2hat(lam,eta) +1.0-c+2.0/eta

def MB_C4(lam,eta):
    return -2.0/eta*MB_C2hat(lam,eta)+2.0/eta

def MB_H1hat(lam,eta,c):
    return MB_C1(lam,eta,c)+MB_C2(lam,eta)*MB_C4(lam,eta)/MB_C3(lam,eta,c)

def MB_H2hat(lam,eta,c):
    return MB_C3(lam,eta,c)+MB_C2(lam,eta)*MB_C4(lam,eta)/MB_C1(lam,eta,c)

def MB_omega_DSA(lam,eta,c):
    I1  = integrate.quad(MB_IG1, -1,1,args=(lam,eta))[0]
    I2  = integrate.quad(MB_IG2, -1,1,args=(lam,eta))[0]
    I12 = integrate.quad(MB_IG12,-1,1,args=(lam,eta))[0]

    C1 = MB_C1(lam,eta,c)
    C2 = MB_C2(lam,eta,)
    C3 = MB_C3(lam,eta,c)
    C4 = MB_C4(lam,eta,)

    H1_hat = MB_H1hat(lam,eta,c)
    H2_hat = MB_H2hat(lam,eta,c)

    H1 = c*(c/2.0*I1-1.0-c/eta*C2/C3*I12)/H1_hat
    H2 = c* (c/2.0/eta*I12+C2/C3*(c/2.0*I2-1))/H1_hat
    H3 = c*(c/2.0*I2-1.0-c/eta/2.0*C4/C1*I12)/H2_hat
    H4 = c* (c/eta*I12+C4/C1*(c/2.0*I1-1))/H2_hat

    a_hat = 1.0
    b_hat = -(c/2.0*I2+H3+c/2.0*I1+H1)
    c_hat = (c/2.0*I2+H3)*(c/2.0*I1+H1)+(c/2.0/eta*I12+H2)*(c/eta*I12+H4)

    D = b_hat**2 - 4.0*a_hat*c_hat

    if D<0:
        o1 = (-b_hat + i*(-D)**0.5)/(2.0*a_hat)
        o2 = (-b_hat - i*(-D)**0.5)/(2.0*a_hat)
    else:
        o1 = (-b_hat + D**0.5)/(2.0*a_hat)
        o2 = (-b_hat - D**0.5)/(2.0*a_hat)

    return o1,o2

def MB_optimize_DSA(lam,eta,c):
    o1,o2 = MB_omega_DSA(lam,eta,c)
    o1 = abs(o1)
    o2 = abs(o2)
    return -max(o1,o2)

def get_rho_BESI(eta,c):
    return rho_BE(eta,c)
def get_rho_MBTDSI(eta,c):
    o1,o2 = MB_omega(1E-10,eta,c)
    return rho_MBTD(eta,c),o1,o2
def get_rho_BEDSA(eta,c):
    x_BE = optimize.fmin(BE_optimize_DSA,0.1,args=(eta,c,1.0),disp=0)
    return BE_omega_DSA(x_BE,eta,c,1.0)
def get_rho_MBTDDSA(eta,c):
    x_MBTD = optimize.fmin(MB_optimize_DSA,0.1,args=(eta,c),disp=0)
    o1,o2 = MB_omega_DSA(x_MBTD,eta,c)
    return max(abs(o1),abs(o2)),o1,o2

# =============================================================================
# Parameters
# =============================================================================

# Kinetics
beta     = 0.0
lam      = 0.08
tgen     = 1E-5

# XS
D        = 1.15
SigmaA   = 0.0263
nuSigmaF = 0.026
SigmaT   = 1.0/3.0/D
v        = 1.0/nuSigmaF/tgen

# Kondimensional
k = nuSigmaF/SigmaA

# Input
S0 = 1.0
S1 = S0
S2 = 0.5*S0
SigmaA1 = SigmaA
SigmaA2 = SigmaA
SigmaT1 = SigmaT
SigmaT2 = SigmaT
SigmaS1 = SigmaT1 - SigmaA1
SigmaS2 = SigmaT2 - SigmaA2

# Initial condition
SigmaA_init = 0.027
k_init      = nuSigmaF/SigmaA_init
phi_init    = S0/(SigmaA_init - nuSigmaF)
C_init      = beta*nuSigmaF/lam*phi_init

# Simulation
eta_list = [10.0]
for eta in eta_list:
    T  = eta/v/SigmaT
    X1 = 500/SigmaT1
    X2 = 500/SigmaT2
    J  = 5000
    N  = 8
    aR = 1.0
    aL = 0.0
    tol = 1E-5
    accelerate = False
    moc        = True
    store      = False
    
    # Report criticality and reactivity
    print('eta',eta)
    k = nuSigmaF/SigmaA1
    c=SigmaS1/SigmaT1
    fac = k+(1-k)*c
    print('c',c)
    print('k',k)
    print('fac',fac)
    
    # Time grids
    K  = 1
    dt = T/K
    
    # Varying parameters
    SigmaA_list = np.ones(K)*SigmaA1
    S_list      = np.ones(K)*S1
    
    phi_BE_spatial = np.zeros([K+1,J])
    C_BE_spatial  = np.zeros([K+1,J])
    psi_BE_spatial = np.zeros([N,J])
    phi_BE_spatial[0][:] = np.ones(J)*phi_init
    C_BE_spatial[0][:]  = np.ones(J)*C_init
    psi_BE_spatial[:,:] = phi_init/2
    
    phi_MB_spatial = np.zeros([K+1,J])
    C_MB_spatial  = np.zeros([K+1,J])
    psi_MB_spatial = np.zeros([N,J])
    phi_MB_spatial[0][:] = np.ones(J)*phi_init
    C_MB_spatial[0][:]  = np.ones(J)*C_init
    psi_MB_spatial[:,:] = phi_init/2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), sharey=True)
    
    [phi_BE_spatial[1:1+int(K/2)], C_BE_spatial[1:1+int(K/2)], psi_BE_spatial, N_iter, et, rho, err,anal] = \
        solvers.BE_trans(psi_BE_spatial,C_BE_spatial[0],v,np.ones(J)*SigmaT1,np.ones(J)*nuSigmaF,np.ones(J)*SigmaS1,beta,lam,np.ones(J)*S1,dt,K,N,X1,J,tol,accelerate,moc)
    print("BE", N_iter)
    #fig = plt.figure(figsize=(4,4))
    ax1.plot(np.arange(1,N_iter-1),rho[1:],'bo',fillstyle="none",label='Numerical')
    rhox = get_rho_BESI(eta,fac)
    ax1.plot(np.arange(1,N_iter-1),np.ones(N_iter-2)*rhox,'-r',label='Theoretical, %.2f'%rhox)
    ax1.set_xlabel(r'Iteration #, ($l$)')
    ax1.set_ylabel(r'$\rho$')
    ax1.grid()
    ax1.legend(loc=4)

    [phi_MB_spatial[1:1+int(K/2)], C_MB_spatial[1:1+int(K/2)], psi_MB_spatial, N_iter, et, rho] = \
        solvers.MB_trans(psi_MB_spatial,C_MB_spatial[0],v,np.ones(J)*SigmaT1,np.ones(J)*nuSigmaF,np.ones(J)*SigmaS1,beta,lam,np.ones(J)*S1,dt,K,N,X1,J,tol,accelerate,moc,store)
    print("MB", N_iter)
    #fig = plt.figure(figsize=(4,4))
    ax2.plot(np.arange(1,N_iter-1),rho[1:],'bo',fillstyle="none",label='Numerical')
    rhox,o1,o2 = get_rho_MBTDSI(eta,fac)
    print(o1,o2)
    ax2.plot(np.arange(1,N_iter-1),np.ones(N_iter-2)*rhox,'-r',label=r'Theoretical, $\|%.2f \pm %.2fi\|$'%(abs(o1.real),abs(o1.imag)))
    ax2.set_xlabel(r'Iteration #, ($l$)')
    ax2.grid()
    ax2.legend()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig("SI.svg",dpi=1200,format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()