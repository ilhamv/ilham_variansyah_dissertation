import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import integrate

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


# SI
eta_list = np.linspace(1E-5,10.0,100); eta_list = np.flip(eta_list)
#eta_list = np.logspace(-5,5,100); eta_list = np.flip(eta_list)
c = 1.0

plt.plot(eta_list,rho_BE(eta_list,c,1.0),'bo-',label="BE-SI",markevery=10)
#plt.plot(eta_list,rho_BE(eta_list,c,2.0),'gD-',label="CN-SI",markevery=10)
#plt.plot(eta_list,rho_BE(eta_list,c,4.0),'m^-',label="TR/BDF2-SI(1st)",markevery=10)
#plt.plot(eta_list,rho_BE(eta_list,c,3.0),'cv-',label="TR/BDF2-SI(2nd)",markevery=10)
plt.plot(eta_list,rho_MBTD(eta_list,c),'rs--',label="MBTD-SI",markevery=10)

# DSA
DSA_BE         = []
DSA_CN         = []
DSA_TRBDF2_1st = []
DSA_TRBDF2_2nd = []
DSA_MBTD       = []

x_BE         = 0.0
x_CN         = 0.0
x_TRBDF2_1st = 0.0
x_TRBDF2_2nd = 0.0
x_MBTD       = 0.0

for eta in eta_list:
    x_BE = optimize.fmin(BE_optimize_DSA,x_BE,args=(eta,c,1.0),disp=0)
    DSA_BE.append(BE_omega_DSA(x_BE,eta,c,1.0))
    #x_CN = optimize.fmin(BE_optimize_DSA,x_CN,args=(eta,c,2.0),disp=0)
    #DSA_CN.append(BE_omega_DSA(x_CN,eta,c,2.0))
    #x_TRBDF2_1st = optimize.fmin(BE_optimize_DSA,x_TRBDF2_1st,args=(eta,c,4.0),disp=0)
    #DSA_TRBDF2_1st.append(BE_omega_DSA(x_TRBDF2_1st,eta,c,4.0))
    #x_TRBDF2_2nd = optimize.fmin(BE_optimize_DSA,x_TRBDF2_2nd,args=(eta,c,3.0),disp=0)
    #DSA_TRBDF2_2nd.append(BE_omega_DSA(x_TRBDF2_2nd,eta,c,3.0))
    x_MBTD = optimize.fmin(MB_optimize_DSA,x_MBTD,args=(eta,c),disp=0)
    o1,o2 = MB_omega_DSA(x_MBTD,eta,c)
    o1 = abs(o1)
    o2 = abs(o2)
    DSA_MBTD.append(max(o1,o2))
    o1,o2 = MB_omega(0.0,eta,c)

plt.plot(eta_list,DSA_BE,'bo-',label="BE-DSA",markevery=10,fillstyle='none')
#plt.plot(eta_list,DSA_CN,'gD--',fillstyle='none',label="CN-DSA",markevery=10)
#plt.plot(eta_list,DSA_TRBDF2_1st,'m^--',fillstyle='none',label="TR/BDF2-DSA(1st)",markevery=10)
#plt.plot(eta_list,DSA_TRBDF2_2nd,'cv--',fillstyle='none',label="TR/BDF2-DSA(2nd)",markevery=10)
plt.plot(eta_list,DSA_MBTD,'rs--',label="MBTD-DSA",markevery=10,fillstyle='none')

plt.xlabel(r'$\eta$')
plt.ylabel(r'$\rho$')
plt.legend()
plt.grid()
#plt.xscale('log')
#plt.yscale('log')
plt.savefig("rho.svg",dpi=1200,format='svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

lam_list = np.linspace(1E-5,20.0,200)
eta = 1.0
c = 1.0

omega_BE = []
omega_DSA_BE = []
omega_CN = []
omega_DSA_CN = []
omega_TBDF1 = []
omega_DSA_TBDF1 = []
omega_TBDF2 = []
omega_DSA_TBDF2 = []
omega_MBTD = []
omega_DSA_MBTD = []

for lam in lam_list:
    omega_BE.append(BE_omega(lam,eta,c,1.0))
    omega_DSA_BE.append(BE_omega_DSA(lam,eta,c,1.0))
    #omega_CN.append(BE_omega(lam,eta,c,2.0))
    #omega_DSA_CN.append(BE_omega_DSA(lam,eta,c,2.0))
    #omega_TBDF1.append(BE_omega(lam,eta,c,4.0))
    #omega_DSA_TBDF1.append(BE_omega_DSA(lam,eta,c,4.0))
    #omega_TBDF2.append(BE_omega(lam,eta,c,3.0))
    #omega_DSA_TBDF2.append(BE_omega_DSA(lam,eta,c,3.0))
    o1,o2 = MB_omega(lam,eta,c)
    o1 = abs(o1)
    o2 = abs(o2) 
    omega_MBTD.append(max(o1,o2))
    o1,o2 = MB_omega_DSA(lam,eta,c)
    o1 = abs(o1)
    o2 = abs(o2) 
    omega_DSA_MBTD.append(max(o1,o2))

plt.plot(lam_list,omega_BE,'bo-',label="BE-SI",markevery=20)
#plt.plot(lam_list,omega_CN,'gD-',label="CN-SI",markevery=20)
#plt.plot(lam_list,omega_TBDF1,'m^-',label="TR/BDF2-SI(1st)",markevery=20)
#plt.plot(lam_list,omega_TBDF2,'cv-',label="TR/BDF2-SI(2nd)",markevery=20)
plt.plot(lam_list,omega_MBTD,'rs--',label="MBTD-SI",markevery=20)

plt.plot(lam_list,omega_DSA_BE,'bo-',label="BE-DSA",markevery=20,fillstyle='none')
#plt.plot(lam_list,omega_DSA_CN,'gD--',label="CN-DSA",markevery=20,fillstyle='none')
#plt.plot(lam_list,omega_DSA_TBDF1,'m^--',label="TR/BDF2-DSA(1st)",markevery=20,fillstyle='none')
#plt.plot(lam_list,omega_DSA_TBDF2,'cv--',label="TR/BDF2-DSA(2nd)",markevery=20,fillstyle='none')
plt.plot(lam_list,omega_DSA_MBTD,'rs--',label="MBTD-DSA",markevery=20,fillstyle='none')
plt.grid()
plt.legend()
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\|\theta(\omega)\|$')
plt.savefig("omega.svg",dpi=1200,format='svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()

