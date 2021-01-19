import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys, os
import platform
import threading
import multiprocessing

sys.path.insert(1, os.path.realpath(os.path.pardir))
import solvers


# =============================================================================
# Report spec
# =============================================================================

print("Computation specs:")
print("  ",platform.machine())
print("  ",platform.version())
print("  ",platform.platform())
print("  ",platform.system())
print("  ",platform.processor())
#print("  ",platform.uname())
print('  ','NumPy ',np.version.version)
print('  ','SciPy ',sp.version.version)
print('  ','thread %i'%threading.active_count())
print('  ','process %i\n'%multiprocessing.cpu_count())

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

# Diffusion model
X   = 900/SigmaT
J   = 1800
aR  = 1.0
aL  = 0.0
tol = 1E-5
T   = T1+T2

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
# Preliminary verification
# =============================================================================

# Analytical solution
N  = 10001
t1 = np.linspace(0,T1,N)
t2 = np.linspace(T1,T1+T2,N)
t2 = np.delete(t2,0)
tx = np.concatenate((t1,t2))
t2 = t2 - t1[-1]

phix = np.zeros(2*N-1)
Cx   = np.zeros(2*N-1)

[phix[:N], Cx[:N]] = solvers.analytical(phi0,C0,v,SigmaA1,nuSigmaF,beta,lam,S1,t1)
[phix[N:], Cx[N:]] = solvers.analytical(phix[N-1],Cx[N-1],v,SigmaA2,nuSigmaF,beta,lam,S2,t2)


# Numerical solutions
N  = 6
dt = (T1+T2)/N
tn = np.linspace(0.0,T1+T2,N+1)

SigmaA_list = np.ones(N)*SigmaA1
SigmaA_list[int(N/2):] = np.ones(int(N/2))*SigmaA2
S_list      = np.ones(N)*S1
S_list[int(N/2):] = np.ones(int(N/2))*S2

# Infinite
[phi_BE, C_BE] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)
[phi_MB, C_MB] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)

# BE Diffusion
phi_BE_spatial = np.zeros([N+1,J])
C_BE_spatial  = np.zeros([N+1,J])
phi_BE_spatial[0][:] = np.ones(J)*phi0
C_BE_spatial[0][:]  = np.ones(J)*C0

[phi_BE_spatial[1:1+int(N/2)], C_BE_spatial[1:1+int(N/2)], et] = \
    solvers.BE_diffusion(phi_BE_spatial[0],C_BE_spatial[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
[phi_BE_spatial[1+int(N/2):], C_BE_spatial[1+int(N/2):], et] = \
    solvers.BE_diffusion(phi_BE_spatial[int(N/2)],C_BE_spatial[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)

# MB Diffusions
phi_MB_spatial = np.zeros([N+1,J])
C_MB_spatial  = np.zeros([N+1,J])
phi_MB_spatial[0][:] = np.ones(J)*phi0
C_MB_spatial[0][:]  = np.ones(J)*C0

phi_MB_spatial_subs = np.zeros([N+1,J])
C_MB_spatial_subs  = np.zeros([N+1,J])
phi_MB_spatial_subs[0][:] = np.ones(J)*phi0
C_MB_spatial_subs[0][:]  = np.ones(J)*C0

phi_MB_spatial_O = np.zeros([N+1,J])
C_MB_spatial_O  = np.zeros([N+1,J])
phi_MB_spatial_O[0][:] = np.ones(J)*phi0
C_MB_spatial_O[0][:]  = np.ones(J)*C0

[phi_MB_spatial[1:1+int(N/2)], C_MB_spatial[1:1+int(N/2)], et] = \
    solvers.MB_diffusion(phi_MB_spatial[0],C_MB_spatial[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
[phi_MB_spatial[1+int(N/2):], C_MB_spatial[1+int(N/2):], et] = \
    solvers.MB_diffusion(phi_MB_spatial[int(N/2)],C_MB_spatial[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)
    
[phi_MB_spatial_subs[1:1+int(N/2)], C_MB_spatial_subs[1:1+int(N/2)], et] = \
    solvers.MB_diffusion_subs(phi_MB_spatial_subs[0],C_MB_spatial_subs[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
[phi_MB_spatial_subs[1+int(N/2):], C_MB_spatial_subs[1+int(N/2):], et] = \
    solvers.MB_diffusion_subs(phi_MB_spatial_subs[int(N/2)],C_MB_spatial_subs[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)

[phi_MB_spatial_O[1:1+int(N/2)], C_MB_spatial_O[1:1+int(N/2)], et] = \
    solvers.MB_diffusion_O(phi_MB_spatial_O[0],C_MB_spatial_O[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol*1E-3)
[phi_MB_spatial_O[1+int(N/2):], C_MB_spatial_O[1+int(N/2):], et] = \
    solvers.MB_diffusion_O(phi_MB_spatial_O[int(N/2)],C_MB_spatial_O[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol*1E-3)

# Verify finite model
plt.plot(phi_MB_spatial[-1]/phi0,'-b',label='Finite model')
plt.plot(phi_MB[-1]*np.ones(J)/phi0,'--r',label='Actual Infinite Problem')
plt.grid()
plt.xlabel(r'$x$, optical thickness')
plt.ylabel(r'$\phi(x,T)/\phi_{init}$')
plt.legend()
plt.show()

# Verify coarse solution
plt.plot(tx*1000,phix/phi0,'k',label='Analytical')
plt.plot(tn*1000,phi_BE/phi0,'--b',fillstyle='none',markevery=int(N/N),label=r'BE_infinite')
plt.plot(tn*1000,phi_MB/phi0,'--r',fillstyle='none',markevery=int(N/N),label=r'MBTD_infinite')
plt.plot(tn*1000,phi_BE_spatial[:,int(-1)]/phi0,'ob',fillstyle='none',markevery=int(N/N),label=r'BE')
plt.plot(tn*1000,phi_MB_spatial[:,int(-1)]/phi0,'sr',fillstyle='none',markevery=int(N/N),label=r'MBTD')
plt.plot(tn*1000,phi_MB_spatial_subs[:,int(-1)]/phi0,'^g',fillstyle='none',markevery=int(N/N),label=r'MBTD (Subs.)')
plt.plot(tn*1000,phi_MB_spatial_O[:,int(-1)]/phi0,'xm',fillstyle='none',markevery=int(N/N),label=r'MBTD (Iter.)')
plt.ylabel(r'$\phi(t)/\phi(0)$')
plt.xlabel(r'$t$, ms')
plt.grid()
plt.legend()
plt.show()


# =============================================================================
# Efficiency
# =============================================================================

N_list  = np.array([2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000])
dt_list = T/N_list
NN = len(N_list)

err_BE  = np.zeros(NN)
err_MB  = np.zeros(NN)

errS_BE      = np.zeros(NN)
errS_MB      = np.zeros(NN)
errS_MB_subs = np.zeros(NN)
errS_MB_O    = np.zeros(NN)

et_BE      = np.zeros(NN)
et_MB      = np.zeros(NN)
et_MB_subs = np.zeros(NN)
et_MB_O    = np.zeros(NN)

for n in range(NN):
    N = N_list[n]
    print(N)
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
    # Numerical infinite
    # =========================================================================

    # Varying parameters
    SigmaA_list = np.ones(N)*SigmaA1
    SigmaA_list[int(N/2):] = np.ones(int(N/2))*SigmaA2
    S_list      = np.ones(N)*S1
    S_list[int(N/2):] = np.ones(int(N/2))*S2
    
    # Solve diffusion
    [phi_BE, C_BE] = solvers.BE_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)
    [phi_MB, C_MB] = solvers.MB_infinite(phi0,C0,v,SigmaA_list,nuSigmaF,beta,lam,S_list,dt,N)
    
    err_BE[n] = np.linalg.norm(np.divide((phi_BE[-1]-phix[-1]),phix[-1]))*100
    err_MB[n] = np.linalg.norm(np.divide((phi_MB[-1]-phix[-1]),phix[-1]))*100
    
    # =========================================================================
    # Diffusion
    # =========================================================================
    
    # BE
    phi_BE_spatial = np.zeros([N+1,J])
    C_BE_spatial  = np.zeros([N+1,J])
    phi_BE_spatial[0][:] = np.ones(J)*phi0
    C_BE_spatial[0][:]  = np.ones(J)*C0
    
    [phi_BE_spatial[1:1+int(N/2)], C_BE_spatial[1:1+int(N/2)],etBE1] = \
        solvers.BE_diffusion(phi_BE_spatial[0],C_BE_spatial[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
    [phi_BE_spatial[1+int(N/2):], C_BE_spatial[1+int(N/2):],etBE2] = \
        solvers.BE_diffusion(phi_BE_spatial[int(N/2)],C_BE_spatial[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)

    # MB
    phi_MB_spatial = np.zeros([N+1,J])
    C_MB_spatial  = np.zeros([N+1,J])
    phi_MB_spatial[0][:] = np.ones(J)*phi0
    C_MB_spatial[0][:]  = np.ones(J)*C0
    
    phi_MB_spatial_subs = np.zeros([N+1,J])
    C_MB_spatial_subs  = np.zeros([N+1,J])
    phi_MB_spatial_subs[0][:] = np.ones(J)*phi0
    C_MB_spatial_subs[0][:]  = np.ones(J)*C0

    phi_MB_spatial_O = np.zeros([N+1,J])
    C_MB_spatial_O  = np.zeros([N+1,J])
    phi_MB_spatial_O[0][:] = np.ones(J)*phi0
    C_MB_spatial_O[0][:]  = np.ones(J)*C0
    
    [phi_MB_spatial[1:1+int(N/2)], C_MB_spatial[1:1+int(N/2)], etMB1] = \
        solvers.MB_diffusion(phi_MB_spatial[0],C_MB_spatial[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
    [phi_MB_spatial[1+int(N/2):], C_MB_spatial[1+int(N/2):], etMB2] = \
        solvers.MB_diffusion(phi_MB_spatial[int(N/2)],C_MB_spatial[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)
        
    [phi_MB_spatial_subs[1:1+int(N/2)], C_MB_spatial_subs[1:1+int(N/2)], etMB_subs1] = \
        solvers.MB_diffusion_subs(phi_MB_spatial_subs[0],C_MB_spatial_subs[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol)
    [phi_MB_spatial_subs[1+int(N/2):], C_MB_spatial_subs[1+int(N/2):], etMB_subs2] = \
        solvers.MB_diffusion_subs(phi_MB_spatial_subs[int(N/2)],C_MB_spatial_subs[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol)
        
    [phi_MB_spatial_O[1:1+int(N/2)], C_MB_spatial_O[1:1+int(N/2)], etMB_O1] = \
        solvers.MB_diffusion_O(phi_MB_spatial_O[0],C_MB_spatial_O[0],v,np.ones(J)*SigmaA1,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S1,dt,int(N/2),X,J,aR,aL,tol*1E-3)
    [phi_MB_spatial_O[1+int(N/2):], C_MB_spatial_O[1+int(N/2):], etMB_O2] = \
        solvers.MB_diffusion_O(phi_MB_spatial_O[int(N/2)],C_MB_spatial_O[int(N/2)],v,np.ones(J)*SigmaA2,np.ones(J)*nuSigmaF,np.ones(J)*D,beta,lam,np.ones(J)*S2,dt,int(N/2),X,J,aR,aL,tol*1E-3)
    
    # Record
    err_BE[n] = np.linalg.norm(np.divide((phi_BE[-1]-phix[-1]),phix[-1]))*100
    errS_BE[n] = np.linalg.norm(np.divide((phi_BE_spatial[-1,int(-1)]-phix[-1]),phix[-1]))*100

    err_MB[n] = np.linalg.norm(np.divide((phi_MB[-1]-phix[-1]),phix[-1]))*100
    errS_MB[n] = np.linalg.norm(np.divide((phi_MB_spatial[-1,int(-1)]-phix[-1]),phix[-1]))*100
    errS_MB_subs[n] = np.linalg.norm(np.divide((phi_MB_spatial_subs[-1,int(-1)]-phix[-1]),phix[-1]))*100
    errS_MB_O[n] = np.linalg.norm(np.divide((phi_MB_spatial_O[-1,int(-1)]-phix[-1]),phix[-1]))*100
    
    et_BE[n] = etBE1+etBE2
    et_MB[n] = etMB1+etMB2
    et_MB_subs[n] = etMB_subs1+etMB_subs2
    et_MB_O[n] = etMB_O1+etMB_O2
    
    # Store
    a = np.asarray([ dt_list,err_BE,errS_BE,et_BE ])
    np.savetxt("diffusion_kp1_BE.csv", a, delimiter=",")

    a = np.asarray([ dt_list,err_MB,errS_MB,et_MB ])
    np.savetxt("diffusion_kp1_MB.csv", a, delimiter=",")

    a = np.asarray([ dt_list,err_MB,errS_MB_subs,et_MB_subs ])
    np.savetxt("diffusion_kp1_MB_subs.csv", a, delimiter=",")
    
    a = np.asarray([ dt_list,err_MB,errS_MB_O,et_MB_O ])
    np.savetxt("diffusion_kp1_MB_O.csv", a, delimiter=",")