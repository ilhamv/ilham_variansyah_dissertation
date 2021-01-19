import numpy as np
import matplotlib.pyplot as plt


# Load
accelerate = True
moc        = False

name = "transport_kp1"
if accelerate:
    name += "_DSA"
else:
    name += "_SI"
if moc:
    name += "_moc"
else:
    name += "_sndd"

BE      = np.genfromtxt(name+'_BE.csv', delimiter=',')
MB      = np.genfromtxt(name+'_MB.csv', delimiter=',')
if moc:
    MB_str = np.genfromtxt(name+'_str_MB.csv', delimiter=',')

dt_list = BE[0,:]
err_BE = BE[1,:]
errS_BE = BE[2,:]
et_BE = BE[3,:]
dt_list *= 1000

err_MB = MB[1,:]
errS_MB = MB[2,:]
et_MB = MB[3,:]

if moc:
    err_MB_str = MB_str[1,:]
    errS_MB_str = MB_str[2,:]
    et_MB_str = MB_str[3,:]

# Verify error
fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,err_BE,'-b',fillstyle='none',label='BE_infinite')
plt.plot(dt_list,errS_BE,'ob',fillstyle='none',label='BE')
plt.plot(dt_list,err_MB,'-r',fillstyle='none',label='MBTD_infinite')
plt.plot(dt_list,errS_MB,'sr',fillstyle='none',label='MBTD')
if moc:
    plt.plot(dt_list,errS_MB_str,'Dg',fillstyle='none',label='MBTD w/ storage')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Relative error, %')
plt.show()

# Runtime
fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,et_BE,'-ob',fillstyle='none',label='BE')
plt.plot(dt_list,et_MB,'-rs',fillstyle='none',label='MBTD')
if moc:
    plt.plot(dt_list,et_MB_str,'--gD',fillstyle='none',label='MBTD w/ storage')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Runtime, s')
plt.savefig(name+'_runtime.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

# Effeciency
fig = plt.figure(figsize=(4,4))
plt.plot(errS_BE,et_BE,'-ob',fillstyle='none',label='BE')
plt.plot(errS_MB,et_MB,'-rs',fillstyle='none',label='MBTD')
if moc:
    plt.plot(errS_MB_str,et_MB_str,'--gD',fillstyle='none',label='MBTD w/ storage')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Relative error, %')
plt.ylabel(r'Runtime, s')
plt.savefig(name+'_eff.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

# Relative runtime
plt.plot(dt_list,et_MB/et_BE,'k')
plt.xscale('log')
plt.xlabel(r'$\Delta t$, s')
plt.ylabel(r'Relative runtime (MB/BE)')
plt.grid()
plt.show()