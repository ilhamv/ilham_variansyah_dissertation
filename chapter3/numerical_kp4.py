import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(1, os.path.realpath(os.path.pardir))
import solvers


# =============================================================================
# Simulation parameters
# =============================================================================

# Kinetics
beta     = 0.0065
lam      = 0.4
tgen     = 1E-5

# Initial XS
D        = 1.15
nuSigmaF = 0.026
SigmaT   = 1.0/3.0/D
v        = 1.0/nuSigmaF/tgen
SigmaA0  = nuSigmaF
S0       = 0.0
k0       = nuSigmaF/SigmaA0

# Part 1
SigmaA1 = 0.0258
S1      = S0
T1      = 0.018

# Part 2
SigmaA2 = SigmaA1*1.009
S2      = S0
T2      = T1
T       = T1+T2

# Report criticality and reactivity
k0 = nuSigmaF/SigmaA0
k1 = nuSigmaF/SigmaA1
k2 = nuSigmaF/SigmaA2
print('k0',k0)
print('rho0',(k0-1)/k0/beta,"$")
print('\nk1',k1)
print('rho1',(k1-1)/k1/beta,"$")
print('\nk2',k2)
print('rho2',(k2-1)/k2/beta,"$")

# Initial condition
phi0 = 1.0
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
N  = 10
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8), sharex=True)
ax1.plot(tx*1000,phix/phi0,'k',label='Analytical')
ax1.plot(tn*1000,phi_BE/phi0,'--ob',fillstyle='none',markevery=int(N/N),label=r'BE')
ax1.plot(tn*1000,phi_CN/phi0,'--sr',fillstyle='none',markevery=int(N/N),label=r'CN')
ax1.plot(tn*1000,phi_MB/phi0,'--Dg',fillstyle='none',markevery=int(N/N),label=r'MBTD')
ax1.set_ylabel(r'$\phi(t)/\phi(0)$')
ax1.grid()
ax1.legend(loc=1)

ax2.plot(tx*1000,Cx/C0,'k',label='Analytical')
ax2.plot(tn*1000,C_BE/C0,'--ob',fillstyle='none',markevery=int(N/N),label=r'BE')
ax2.plot(tn*1000,C_CN/C0,'--sr',fillstyle='none',markevery=int(N/N),label=r'CN')
ax2.plot(tn*1000,C_MB/C0,'--Dg',fillstyle='none',markevery=int(N/N),label=r'MBTD')
ax2.grid()
ax2.set_xlabel(r'$t$, ms')
ax2.set_ylabel(r'$C(t)/C(0)$')
plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.savefig('numerical_kp4.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()


# =============================================================================
# Plots coarse
# =============================================================================

# Time grids
N  = 4
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

plt.plot(tx*1E3,phix/phi0,'k',label='Analytical')
plt.plot(tn*1E3,phi_BE/phi0,'--ob',fillstyle='none',markevery=int(N/N),label=r'BE')
plt.plot(tn*1E3,phi_CN/phi0,'--sr',fillstyle='none',markevery=int(N/N),label=r'CN')
plt.plot(tn*1E3,phi_MB/phi0,'--Dg',fillstyle='none',markevery=int(N/N),label=r'MBTD')
plt.ylabel(r'$\phi(t)/\phi(0)$')
plt.xlabel(r'$t$, ms')
plt.grid()
plt.ylim([-50, 70])
plt.legend(loc=1)
plt.savefig('numerical_coarse_kp4.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
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

dt_list = dt_list*1E3
first = dt_list
second = dt_list**2
fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,err_BE,'-ob',fillstyle='none',label='BE')
plt.plot(dt_list,err_CN,'-sr',fillstyle='none',label='CN')
plt.plot(dt_list,err_MB,'-Dg',fillstyle='none',label='MBTD')
plt.plot(dt_list,first/first[-1]*3E-2,'--k',label='1st order')
plt.plot(dt_list,second/second[-1]*4E-6,':k',label='2nd order')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Relative error, %')
plt.savefig('numerical_accuracy_kp4.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()


# =============================================================================
# Accuracy (delayed source)
# =============================================================================

N_list  = np.array([2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000])
dt_list = T/N_list
NN = len(N_list)
err_BE  = np.zeros(NN)
err_MB  = np.zeros(NN)

err_BEl  = np.zeros(NN)
err_MBl  = np.zeros(NN)

err_BEq  = np.zeros(NN)
err_MBq  = np.zeros(NN)

err_BEn  = np.zeros(NN)
err_MBn  = np.zeros(NN)

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
    [phi_BEl, C_BEl] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'linear')
    [phi_BEq, C_BEq] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'quad')
    [phi_BEn, C_BEn] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'BE+')
    [phi_MB, C_MB] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'standard')
    [phi_MBl, C_MBl] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,'linear')
    [phi_MBq, C_MBq] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,dsource='quad')
    [phi_MBn, C_MBn] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N,dsource='MB+')
    
    err_BE[n] = np.linalg.norm(np.divide((phi_BE[-1]-phix[-1]),phix[-1]))*100
    err_MB[n] = np.linalg.norm(np.divide((phi_MB[-1]-phix[-1]),phix[-1]))*100
    
    err_BEl[n] = np.linalg.norm(np.divide((phi_BEl[-1]-phix[-1]),phix[-1]))*100
    err_MBl[n] = np.linalg.norm(np.divide((phi_MBl[-1]-phix[-1]),phix[-1]))*100

    err_BEq[n] = np.linalg.norm(np.divide((phi_BEq[-1]-phix[-1]),phix[-1]))*100
    err_MBq[n] = np.linalg.norm(np.divide((phi_MBq[-1]-phix[-1]),phix[-1]))*100
    
    err_BEn[n] = np.linalg.norm(np.divide((phi_BEn[-1]-phix[-1]),phix[-1]))*100
    err_MBn[n] = np.linalg.norm(np.divide((phi_MBn[-1]-phix[-1]),phix[-1]))*100

dt_list = dt_list*1E3
first = dt_list
second = dt_list**2
fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,err_BE,'--ob',fillstyle='none')
plt.plot(dt_list,err_BEl,':+b',fillstyle='none')
plt.plot(dt_list,err_BEq,':vb',fillstyle='none')
#plt.plot(dt_list,err_BEn,':+b',fillstyle='none',label='BE (new)')
plt.plot(dt_list,err_MB,'--sr',fillstyle='none')
plt.plot(dt_list,err_MBl,':xr',fillstyle='none')
plt.plot(dt_list,err_MBq,':Dr',fillstyle='none')
plt.plot(dt_list,err_MBn,':*r',fillstyle='none')
plt.plot(dt_list,first/first[-1]*2E-1,'--k',label='1st order')
plt.plot(dt_list,second/second[-1]*1E-3,':k',label='2nd order')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Relative error, %')
plt.savefig('numerical_Dsource_kp4.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()