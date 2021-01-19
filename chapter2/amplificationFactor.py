import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Amplification factor of time-stepping methods
# =============================================================================

def A_exact(eta):
    return np.e**-eta
def A_FE(eta):
    return 1.0 - eta

def A_BE(eta):
    return 1.0/(1.0 + eta)

def A_CN(eta):
    return (1.0 - 0.5*eta)/(1.0 + 0.5*eta)

def A_TRBDF2(eta):
    return (12.0 - 5.0*eta)/(12.0 + 7.0*eta + eta**2)

def A_MBTD(eta):
    return 1.0/(1.0 + eta + 0.5*eta**2)

# =============================================================================
# Plot
# =============================================================================

eta_list = np.linspace(0.0,20.0,1001)
plt.plot(eta_list,A_exact(eta_list),'-k',label='Exact',markevery=100,fillstyle='none')
plt.plot(eta_list,A_FE(eta_list),'--y',label='FE',markevery=100,fillstyle='none')
plt.plot(eta_list,A_BE(eta_list),'--ob',label='BE',markevery=100,fillstyle='none')
plt.plot(eta_list,A_CN(eta_list),'--sr',label='CN',markevery=100,fillstyle='none')
plt.plot(eta_list,A_TRBDF2(eta_list),'--^m',label='TR-BDF2',markevery=100,fillstyle='none')
plt.plot(eta_list,A_MBTD(eta_list),'--gD',label='MBTD',markevery=100,fillstyle='none')
plt.xlim([-1.0,21.0])
plt.ylim([-1.1,1.1])
plt.grid()
plt.legend()
plt.xlabel(r'$\eta$')
plt.ylabel(r'$A(\eta)$')
plt.savefig('amplificationFactor.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()