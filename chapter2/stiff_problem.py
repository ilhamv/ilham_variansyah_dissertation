import matplotlib.pyplot as plt
import numpy as np


# Problem parameters
a   = 0.5
b   = 20.0
c   = 20.0
B   = b/(a-c)


# =============================================================================
# Analytical solution
# =============================================================================

# Time grids
t = np.linspace(0.0,2.0,10000)

def x_exact(t):
    return np.e**(-c*t)

def y_exact(t):
    return B*(np.e**(-c*t) - np.e**(-a*t))


# =============================================================================
# Numerical solutions
# =============================================================================

# Time grids
t_num = np.linspace(0.0,2.0,11)
dt    = t_num[2] - t_num[1]

# FE
x    = 1.0
y    = 0.0
y_FE = [y]
for i in range(len(t_num)-1):
    # New x
    x = (1.0 - dt*c)*x

    # New y
    y = (1.0 - dt*a)*y + dt*b*x

    y_FE.append(y)

# BE
x    = 1.0
y    = 0.0
y_BE = [y]
for i in range(len(t_num)-1):
    # New x
    x = 1.0/(1.0 + dt*c)*x

    # New y
    y = (y + dt*b*x)/(1.0 + dt*a)

    y_BE.append(y)

# CN
x    = 1.0
y    = 0.0
y_CN = [y]
for i in range(len(t_num)-1):
    # New x
    x_old = x
    x     = (1.0 - 0.5*dt*c)/(1.0 + 0.5*dt*c)*x
    x_avg = 0.5*(x_old + x)

    # New y
    y     = ((1.0 - 0.5*dt*a)*y + dt*b*x_avg)/(1.0 + 0.5*dt*a)

    y_CN.append(y)

# TRBDF2
x    = 1.0
y    = 0.0
y_TRBDF2 = [y]
for i in range(len(t_num)-1):
    # New x
    x_old  = x
    x_half = (1.0 - 0.25*dt*c)/(1.0 + 0.25*dt*c)*x
    x      = (12.0 - 5.0*dt*c)/(12.0 + 7.0*dt*c + (dt*c)**2)*x
    x_avg  = (x_old + x_half + x)/3.0
    x_avg2 = (x_old + x_half)/2.0

    # New y
    y_half = ((1.0 - 0.25*dt*a)*y + 0.5*dt*b*x_avg2)/(1.0 + 0.25*dt*a)
    y      = (dt*b*x_avg + (1.0 - dt*a/3.0)*y - dt*a/3.0*y_half)/(1.0+dt*a/3.0)

    y_TRBDF2.append(y)

# MBTD
x    = 1.0
y    = 0.0
y_MBTD = [y]
for i in range(len(t_num)-1):
    # New x
    x     = 1.0/(1.0 + dt*c + 0.5*(dt*c)**2)*x
    x_avg = (1.0+0.5*dt*c)*x

    # New y
    y     = (y + dt*b*x_avg + dt*0.5*dt*a*b*x) / (1.0 + dt*a + 0.5*(dt*a)**2)

    y_MBTD.append(y)

# =============================================================================
# Plot
# =============================================================================

plt.plot(t,y_exact(t),'k-',label="Exact",fillstyle='none')
plt.plot(t_num,y_FE,'y:',label="FE",fillstyle='none')
plt.plot(t_num,y_BE,'b--o',label="BE",fillstyle='none')
plt.plot(t_num,y_CN,'r--s',label="CN",fillstyle='none')
plt.plot(t_num,y_TRBDF2,'^--m',label="TR-BDF2",fillstyle='none')
plt.plot(t_num,y_MBTD,'g--D',label="MBTD",fillstyle='none')
plt.ylim([-0.05,1.2])
plt.grid()
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.savefig('stiff_problem.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()