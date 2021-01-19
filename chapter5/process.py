import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import scipy as sp
import scipy.integrate


# Reference
with h5py.File('CN/td_2000.h5','r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    rho   = f['rho'][()]

exact = sp.integrate.simps(power,t)
ax1 = plt.subplot()
l1=plt.plot(t,power,'-k',label=r"Ref.")

exact_peak = max(power)
exact_peakt = t[np.argmax(power)]

# MBTD - 100
K = 100
dt = 1.0/K
mev = max(int(K/10),1)
with h5py.File('MBTD/td_%i.h5'%K,'r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    te    = f['te'][()]
l2 = plt.plot(t,power,'--sr',markevery=mev,label=r"MBTD (%i, %.0f s)"%(K,te),fillstyle='none')

# BE - 1000
K = 1000
dt = 1.0/K
mev = max(int(K/10),1)
with h5py.File('BE/td_%i.h5'%K,'r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    te    = f['te'][()]
l3=plt.plot(t,power,'-.^g',markevery=mev,label=r"BE (%i, %.0f s)"%(K,te),fillstyle='none')

# BE - 600
K = 600
dt = 1.0/K
mev = max(int(K/10),1)
with h5py.File('BE/td_%i.h5'%K,'r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    te    = f['te'][()]
l4=plt.plot(t,power,'-.ob',markevery=mev,label=r"BE (%i, %.0f s)"%(K,te),fillstyle='none')

# BE - 100
K = 100
dt = 1.0/K
mev = max(int(K/10),1)
with h5py.File('BE/td_%i.h5'%K,'r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    te    = f['te'][()]
l5=plt.plot(t,power,'-.vm',markevery=mev,label=r"BE (%i, %.0f s)"%(K,te),fillstyle='none')


plt.ylabel('Power (W/cm)')
plt.xlabel('Time (s)')
plt.grid()
ax2 = ax1.twinx()
with h5py.File('CN/td_2000.h5','r') as f:
    t     = f['t'][()]
    power = f['pow'][()]
    rho   = f['rho'][()]
l6=ax2.plot(t,rho*100.0,':k',alpha=0.5,label='Reactivity')
ax2.set_ylabel('Reactivity (%)')
ln = l1+l2+l3+l4+l5+l6
lab = [l.get_label() for l in ln]
plt.legend(ln, lab, loc=6,bbox_to_anchor=(0.01,0.36))
plt.savefig('power.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

K_list = [20,40,60,80,100,200,400,600,800,1000]
dt = []
BE = []
MBTD = []
t_BE = []
t_MBTD = []

BE_peak = []
MBTD_peak = []
BE_peakt = []
MBTD_peakt = []

for K in K_list:
    dt.append(1.0/K)
    
    # BE
    with h5py.File('BE/td_%i.h5'%K,'r') as f:
        t     = f['t'][()]
        power = f['pow'][()]
        te    = f['te'][()]
    #plt.plot(power,t,'--ob',markevery=mev)
    BE.append(abs(exact-sp.integrate.simps(power,t))/exact*100)
    BE_peak.append(abs(exact_peak-max(power))/exact_peak*100)
    BE_peakt.append(abs(exact_peakt-t[np.argmax(power)])/exact_peakt*100)
    t_BE.append(te)

    # MBTD - OS
    '''
    with h5py.File('MBTDOS/td_%i.h5'%K,'r') as f:
        t     = f['t'][()]
        power = f['pow'][()]
    plt.plot(power,t,'--*m',markevery=mev)
    '''

    # MBTD
    with h5py.File('MBTD/td_%i.h5'%K,'r') as f:
        t     = f['t'][()]
        power = f['pow'][()]
        te    = f['te'][()]
    #plt.plot(power,t,'--sr',markevery=mev)
    MBTD.append(abs(exact-sp.integrate.simps(power,t))/exact*100)
    MBTD_peak.append(abs(exact_peak-max(power))/exact_peak*100)
    MBTD_peakt.append(abs(exact_peakt-t[np.argmax(power)])/exact_peakt*100)
    t_MBTD.append(te)
    #plt.show()

# Order
plt.plot(dt,BE,'b-o',label='BE')
plt.plot(dt,MBTD,'r-s',label='MBTD')
plt.xlabel(r'$\Delta t$ (s)')
plt.ylabel('Relative error of total power (%)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('order.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# Order
plt.plot(dt,BE_peak,'b-o',label='BE')
plt.plot(dt,MBTD_peak,'r-s',label='MBTD')
plt.xlabel(r'$\Delta t$ (s)')
plt.ylabel('Relative error of peak power (%)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('order_peak.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# Order
plt.plot(dt,BE_peakt,'b-o',label='BE')
plt.plot(dt,MBTD_peakt,'r-s',label='MBTD')
plt.xlabel(r'$\Delta t$ (s)')
plt.ylabel('Relative error of time at peak power (%)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('order_peakt.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# time
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot()
ax1.plot(dt,t_BE,'b-o',label='BE',fillstyle='none')
ax1.plot(dt,t_MBTD,'r-s',label='MBTD',fillstyle='none')
ax1.grid()
ax1.legend(loc=1)
ax1.set_xlabel(r'$\Delta t$ (s)')
ax1.set_ylabel('Runtime (s)')
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.savefig('time.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# time vs error
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot()
ax1.plot(BE,t_BE,'b-o',label='BE',fillstyle='none')
ax1.plot(MBTD,t_MBTD,'r-s',label='MBTD',fillstyle='none')
plt.xlabel('Relative error of total power (%)')
plt.ylabel('Runtime (s)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('time_err.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# time vs error
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot()
ax1.plot(BE_peak,t_BE,'b-o',label='BE',fillstyle='none')
ax1.plot(MBTD_peak,t_MBTD,'r-s',label='MBTD',fillstyle='none')
plt.xlabel('Relative error of peak power (%)')
plt.ylabel('Runtime (s)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('time_err_peak.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# time vs error
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot()
ax1.plot(BE_peakt,t_BE,'b-o',label='BE',fillstyle='none')
ax1.plot(MBTD_peakt,t_MBTD,'r-s',label='MBTD',fillstyle='none')
plt.xlabel('Relative error of time at peak power (%)')
plt.ylabel('Runtime (s)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('time_err_peakt.svg', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()