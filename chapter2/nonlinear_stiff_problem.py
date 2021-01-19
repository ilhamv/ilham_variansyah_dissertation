import matplotlib.pyplot as plt
import numpy as np


# Parameters
lam = -1E6
eta = 1.5

# Analytical solution
def exact(t):
    return np.e**(lam*t)*(eta-1) + np.cos(t)
t = np.linspace(0.0,3.0,int(1E7))


# =============================================================================
# Numerical solutions
# =============================================================================
dt    = 0.1
t_num = np.arange(0.0,3.0+dt,dt)

# BE
u0   = eta
t0   = 0.0
u_BE = [eta]
for i in range(len(t_num)-1):
    t1 = t0 + dt
    u1 = (u0 - dt*(lam*np.cos(t1)+np.sin(t1)))/(1.0 - dt*lam)
    u_BE.append(u1)
    u0 = u1
    t0 = t1

# CN
u0   = eta
t0   = 0.0
u_CN = [eta]
for i in range(len(t_num)-1):
    t1 = t0 + dt
    u1 = (u0 - dt/2*(lam*np.cos(t1)+np.sin(t1)) + dt/2*(lam*(u0-np.cos(t0))-np.sin(t0)))/(1.0 - dt/2*lam)
    u_CN.append(u1)
    u0 = u1
    t0 = t1

# MB
u0   = eta
t0   = 0.0
u_MBTD = [eta]
for i in range(len(t_num)-1):
    t1 = t0 + dt
    t12 = t0 + dt/2
    u1 = (u0 + dt*(lam*dt/2*(lam*np.cos(t1)+np.sin(t1))-lam*np.cos(t12)-np.sin(t12)))/(1.0 - dt*lam*(1-dt/2*lam))
    u_MBTD.append(u1)
    u0 = u1
    t0 = t1
    

# =============================================================================
# Plot
# =============================================================================

plt.plot(t,exact(t),'k-',label="Exact",fillstyle='none')
plt.plot(t_num,u_BE,'b--o',label="BE",fillstyle='none')
plt.plot(t_num,u_CN,'r--s',label="CN",fillstyle='none')
plt.plot(t_num,u_MBTD,'g:D',label="MBTD",fillstyle='none')
plt.xlim([0.0,3])
plt.ylim([-1.5,1.5])
plt.grid()
plt.legend(loc=1)
plt.xlabel(r'$t$')
plt.ylabel(r'$u(t)$')
plt.savefig('nonlinear_stiff_problem.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()