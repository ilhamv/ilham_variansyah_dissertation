import numpy as np
import matplotlib.pyplot as plt


# Load
BE      = np.genfromtxt('diffusion_kp1_BE.csv', delimiter=',')
MB      = np.genfromtxt('diffusion_kp1_MB.csv', delimiter=',')
MB_subs = np.genfromtxt('diffusion_kp1_MB_subs.csv', delimiter=',')
MB_O    = np.genfromtxt('diffusion_kp1_MB_O.csv', delimiter=',')

dt_list = BE[0,:]
err_BE = BE[1,:]
errS_BE = BE[2,:]
et_BE = BE[3,:]
dt_list *= 1000

err_MB = MB[1,:]
errS_MB = MB[2,:]
et_MB = MB[3,:]

err_MB_subs = MB_subs[1,:]
errS_MB_subs = MB_subs[2,:]
et_MB_subs = MB_subs[3,:]

err_MB_O = MB_O[1,:]
errS_MB_O = MB_O[2,:]
et_MB_O = MB_O[3,:]

# Verify error
fig = plt.figure(figsize=(4,4))
plt.plot(dt_list,err_BE,'-b',fillstyle='none',label='BE_infinite')
plt.plot(dt_list,errS_BE,'ob',fillstyle='none',label='BE')
plt.plot(dt_list,err_MB,'-r',fillstyle='none',label='MBTD_infinite')
plt.plot(dt_list,errS_MB,'sr',fillstyle='none',label='MBTD')
plt.plot(dt_list,errS_MB_subs,'^g',fillstyle='none',label='MBTD (Subs.)')
plt.plot(dt_list,errS_MB_O,'xm',fillstyle='none',label='MBTD (iter.)')
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
plt.plot(dt_list,et_MB_subs,'--^g',fillstyle='none',label='MBTD (Subs.)')
plt.plot(dt_list,et_MB_O,'--xm',fillstyle='none',label='MBTD (Iter.)')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$, ms')
plt.ylabel(r'Runtime, s')
plt.savefig('diffusion_runtime_kp1.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

# Effeciency
fig = plt.figure(figsize=(4,4))
plt.plot(errS_BE,et_BE,'-ob',fillstyle='none',label='BE')
plt.plot(errS_MB,et_MB,'-rs',fillstyle='none',label='MBTD')
plt.plot(errS_MB_subs,et_MB_subs,'--^g',fillstyle='none',label='MBTD (Subs.)')
plt.plot(errS_MB_subs,et_MB_O,'--xm',fillstyle='none',label='MBTD (Iter.)')
plt.legend(loc=0)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Relative error, %')
plt.ylabel(r'Runtime, s')
plt.savefig('diffusion_eff_kp1.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()