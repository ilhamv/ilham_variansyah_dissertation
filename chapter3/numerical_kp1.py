import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))
import solvers


# =============================================================================
# Simulation parameters
# =============================================================================

# Kinetics
beta     = 0.0
lam      = 0.08
tgen     = 1E-5

# Initial XS
D        = 1.15
nuSigmaF = 0.026
SigmaT   = 1.0/3.0/D
v        = 1.0/nuSigmaF/tgen
SigmaA0  = 0.027
S0       = 1.0
k0       = nuSigmaF/SigmaA0

# Part 1
SigmaA1 = 0.0263
S1      = S0
T1      = 8E-3

# Part 2
SigmaA2 = SigmaA1
S2      = 0.5*S1
T2      = T1
T       = T1+T2

# Report criticality and reactivity
k0 = nuSigmaF/SigmaA0
k1 = nuSigmaF/SigmaA1
k2 = nuSigmaF/SigmaA2
print('k0',k0)
print('rho0',(k0-1)/k0*100,"%")
print('\nk1',k1)
print('rho1',(k1-1)/k1*100,"%")
print('\nk2',k2)
print('rho2',(k2-1)/k2*100,"%")

# Initial condition
phi0 = S0/(SigmaA0 - nuSigmaF)
C0   = beta*nuSigmaF/lam*phi0


# =============================================================================
# Analytical solutions
# =============================================================================

# Time grids
N  = 10001
t1 = np.linspace(0,T1,N)
t2 = np.linspace(T1,T1+T2,N)
t2 = np.delete(t2,0)
tx = np.concatenate((t1,t2))
t2 = t2 - t1[-1]

# Allocate solution
phix = np.zeros(2*N-1)
Cx   = np.zeros(2*N-1)

# Solve
[phix[:N], Cx[:N]] = solvers.analytical(phi0,C0,v,SigmaA1,nuSigmaF,beta,lam,S1,t1)
[phix[N:], Cx[N:]] = solvers.analytical(phix[N-1],Cx[N-1],v,SigmaA2,nuSigmaF,beta,lam,S2,t2)


# =============================================================================
# Numerical solutions
# =============================================================================

# Time grids
N  = 6
dt = (T1+T2)/N
tn = np.linspace(0.0,T1+T2,N+1)

# Varying parameters
SigmaA_list = np.ones(N)*SigmaA1
SigmaA_list[int(N/2):] = np.ones(int(N/2))*SigmaA2
S_list      = np.ones(N)*S1
S_list[int(N/2):] = np.ones(int(N/2))*S2

# Solve
[phi_BE, C_BE] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'standard')
[phi_CN, C_CN] = solvers.CN_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)
[phi_MB, C_MB] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,dsource='standard')


# =============================================================================
# Plots
# =============================================================================

plt.plot(tx*1000,phix/phi0,'k',label='Analytical')
plt.plot(tn*1000,phi_BE/phi0,'--ob',fillstyle='none',markevery=int(N/N),label=r'BE')
plt.plot(tn*1000,phi_CN/phi0,'--sr',fillstyle='none',markevery=int(N/N),label=r'CN')
plt.plot(tn*1000,phi_MB/phi0,'--Dg',fillstyle='none',markevery=int(N/N),label=r'MBTD')
plt.ylabel(r'$\phi(t)/\phi(0)$')
plt.xlabel(r'$t$, ms')
plt.grid()
plt.legend()
plt.savefig('numerical_kp1.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()


# =============================================================================
# Accuracy
# =============================================================================

N_list  = np.array([2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000])
dt_list = T/N_list
NN = len(N_list)
err_BE  = np.zeros(NN)
err_CN  = np.zeros(NN)
err_MB  = np.zeros(NN)

for n in range(NN):
    N = N_list[n]
    dt = T/N
    tn = np.linspace(0.0,T,N+1)
    
    # =========================================================================
    # Analytical
    # =========================================================================
    # Time grids
    t1 = np.linspace(0,T/2,int(N/2)+1)
    t2 = np.linspace(T/2,T,int(N/2)+1)
    t2 = np.delete(t2,0)
    tx = np.concatenate((t1,t2))
    t2 = t2 - t1[-1]
    
    # Allocate solution
    phix = np.zeros(N+1)
    Cx   = np.zeros(N+1)
    
    # Solve
    [phix[:int(N/2)+1], Cx[:int(N/2)+1]] = solvers.analytical(phi0,C0,v,SigmaA1,nuSigmaF,beta,lam,S1,t1)
    [phix[int(N/2)+1:], Cx[int(N/2)+1:]] = solvers.analytical(phix[int(N/2)],Cx[int(N/2)],v,SigmaA2,nuSigmaF,beta,lam,S2,t2)    
    
    # =========================================================================
    # Numerical
    # =========================================================================

    # Varying parameters
    SigmaA_list = np.ones(N)*SigmaA1
    SigmaA_list[int(N/2):] = np.ones(int(N/2))*SigmaA2
    S_list      = np.ones(N)*S1
    S_list[int(N/2):] = np.ones(int(N/2))*S2
    
    # Solve numerical
    [phi_BE, C_BE] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'standard')
    [phi_CN, C_CN] = solvers.CN_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)
    [phi_MB, C_MB] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,dsource='standard')
    
    err_BE[n] = np.linalg.norm(np.divide((phi_BE[-1]-phix[-1]),phix[-1]))*100
    err_CN[n] = np.linalg.norm(np.divide((phi_CN[-1]-phix[-1]),phix[-1]))*100
    err_MB[n] = np.linalg.norm(np.divide((phi_MB[-1]-phix[-1]),phix[-1]))*100

dt_list *= 1000
first = dt_list
second = dt_list**2

fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,err_BE,'-ob',fillstyle='none',label='BE')
plt.plot(dt_list,err_CN,'-sr',fillstyle='none',label='CN')
plt.plot(dt_list,err_MB,'-Dg',fillstyle='none',label='MBTD')
plt.plot(dt_list,first/first[-1]*2E-4,'--k',label='1st order')
plt.plot(dt_list,second/second[-1]*2E-7,':k',label='2nd order')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Relative error, %')
plt.savefig('numerical_accuracy_kp1.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()