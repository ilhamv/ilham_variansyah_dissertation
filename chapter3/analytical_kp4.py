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

fig, ax1a = plt.subplots(figsize=(4,4))
ax2a = ax1a.twinx()
ax1b = ax1a.twinx()
ax2b = ax1a.twinx()

ax1a.get_shared_y_axes().join(ax1a, ax1b)
ax2a.get_shared_y_axes().join(ax2a, ax2b)

ticksoff = dict(labelleft=False, labelright=False, left=False, right=False)
ax2a.tick_params(axis="y", **ticksoff)
ax1b.tick_params(axis="y", **ticksoff)

p1, = ax1b.plot(t*1000,phi/phi0,'b',label=r'$\phi(t)/\phi(0)$')
p2, = ax2b.plot(t*1000,C/C0,'--r',label=r'$C(t)/C(0)$')

ax1a.set_xlabel(r'$t$, ms')
ax1a.set_ylabel(r'$\phi(t)/\phi(0)$',color='b')
ax2b.set_ylabel(r'$C(t)/C(0)$',color='r')

ax1a.tick_params(axis='y', colors='b')
ax2b.tick_params(axis='y', colors='r')

ax1a.grid()
ax2a.grid(lw=0.5)

lines = [p1,p2]
ax2b.legend(lines,[l.get_label() for l in lines],loc=2)

ax1a.annotate('', xy=(t[6000]*1000, phi[6000]), xytext=(t[4000]*1000, phi[6000]), 
              arrowprops=dict(arrowstyle="<-", color='k'))
ax2b.annotate('', xy=(t[7000]*1000, C[7000]/C0), xytext=(t[9000]*1000, C[7000]/C0), 
              arrowprops=dict(arrowstyle="<-", color='k'))

plt.savefig('analytical_kp4.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)