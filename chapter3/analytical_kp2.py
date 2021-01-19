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
T1      = 50.0

# Part 2
SigmaA2 = SigmaA1
S2      = 0.5*S1
T2      = T1

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
t  = np.concatenate((t1,t2))
t2 = t2 - t1[-1]

# Allocate solution
phi = np.zeros(2*N-1)
C   = np.zeros(2*N-1)

# Solve
[phi[:N], C[:N]] = solvers.analytical(phi0,C0,v,SigmaA1,nuSigmaF,beta,lam,S1,t1)
[phi[N:], C[N:]] = solvers.analytical(phi[N-1],C[N-1],v,SigmaA2,nuSigmaF,beta,lam,S2,t2)


# =============================================================================
# Plots
# =============================================================================

fig = plt.figure(figsize=(4,4))
plt.plot(t,phi/phi0,'b',label=r'$\phi(t)/\phi(0)$')
plt.plot(t,C/C0,'--r',label=r'$C(t)/C(0)$')
plt.xlabel(r'$t$, s')
plt.grid()
plt.legend()
plt.savefig('analytical_kp2.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)