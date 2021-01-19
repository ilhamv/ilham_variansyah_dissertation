import numpy as np
import matplotlib.pyplot as plt


# Get spectral radius of G = M^-1 dot N
def get_rho(M,N):
    G = (np.linalg.inv(M)).dot(N)
    theta, x = np.linalg.eig(G)
    theta = abs(theta)
    theta = max(theta)
    return theta

# List of eta
eta_list = np.logspace(-1,1,1001)

# Generate results
rho_1 = []
rho_2 = []
rho_3 = []
rho_4 = []
rho_5 = []
for eta in eta_list:
    # 1: Simple lag
    M = np.array([[-1, 1 + 0.5*eta],[eta, 0]])
    N = np.array([[0.0, 0.0],[0, -1]])
    rho_1.append(get_rho(M,N))

    # 2: Parallel Solves
    M = np.array([[eta, 0],[0, 1 + 0.5*eta]])
    N = np.array([[0, -1],[1, 0.0]])
    rho_2.append(get_rho(M,N))
    
    # 3: Substitution with lag
    M = 1 + eta
    N = -0.5*eta**2
    rho_3.append(abs(N/M))
    
    # 4: O(dt^2)
    M = np.array([[2 + eta, 0], [-1, 1 + 0.5*eta]])
    N = np.array([[2, -1], [0.0, 0.0]])
    rho_4.append(get_rho(M,N))

    # 5: O(dt)
    M = np.array([[1 + eta, 0], [-1, 1 + 0.5*eta]])
    N = np.array([[1, -1], [0.0, 0.0]])
    rho_5.append(get_rho(M,N))
    
# =============================================================================
# Plot
# =============================================================================
plt.plot(eta_list,rho_1,'-sb',label="Simple Lag",markevery=100,fillstyle='none')
plt.plot(eta_list,rho_2,'-Dr',label="Parallel Lag",markevery=100,fillstyle='none')
plt.plot(eta_list,rho_3,'-ok',label="Lagged Substitution",markevery=100,fillstyle='none')
plt.plot(eta_list,rho_4,'-^m',label=r"Lagged $O(\Delta t^2)$",markevery=100,fillstyle='none')
plt.plot(eta_list,rho_5,'-vg',label=r"Lagged $O(\Delta t)$",markevery=100,fillstyle='none')
plt.ylim([0.0,1.1])
plt.xscale('log')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\rho (G)$')
plt.grid()
plt.legend(bbox_to_anchor=(.01, 0.9), loc=2, borderaxespad=0.)
plt.savefig('iterative_solve.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()