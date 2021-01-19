import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import scipy as sp
import scipy.interpolate
import h5py
import sys
import os
from matplotlib import cm
from matplotlib import colors
sys.path.insert(1, os.path.realpath(os.path.pardir))


K_list = [2000]
time_list = []
for K in K_list:
    with h5py.File('td_%i.h5'%K,'r') as f:
        phi   = f['phi'][()]
        rho   = f['rho'][()]
        Tf    = f['Tf'][()]
        Tw    = f['Tw'][()]
        t     = f['t'][()]
        power = f['pow'][()]
        time_list.append(f['te'])
    K,J = phi.shape
    T = 1.0
    Z = 360.0
    x = np.linspace(0,Z,J+1)
    x = (x[1:] + x[:-1])/2

    # Sent for figure
    font = {'size': 9}
    matplotlib.rc('font', **font)

    # Setup figure and subplots
    fig = plt.figure(figsize = (12, 8))
    ax01 = plt.subplot2grid((2, 2), (0, 0))
    ax02 = plt.subplot2grid((2, 2), (0, 1))
    ax03 = plt.subplot2grid((2, 2), (1, 0))
    ax04 = plt.subplot2grid((2, 2), (1, 1))
    fig.tight_layout(pad=4.0)

    # Set titles of subplots
    ax01.set_title('Neutron Flux')
    ax02.set_title('Fuel Temperature')
    ax03.set_title('Reactiviy')
    ax04.set_title('Coolant Temperature')

    # set y-limits
    rmax = phi.max(); rmin = phi.min(); r = (rmax-rmin)*0.1
    ax01.set_ylim(rmin-r,rmax+r)
    rmax = Tf.max(); rmin = Tf.min(); r = (rmax-rmin)*0.1
    ax02.set_ylim(rmin-r,rmax+r)
    rmax = rho.max(); rmin = rho.min(); r = (rmax-rmin)*0.1
    ax03.set_ylim(rmin-r,rmax+r)
    rmax = Tw.max(); rmin = Tw.min(); r = (rmax-rmin)*0.1
    ax04.set_ylim(rmin-r,rmax+r)

    # sex x-limits
    ax01.set_xlim(-10.0,370.0)
    ax02.set_xlim(-10.0,370.0)
    ax03.set_xlim(0.0,t[-1])
    ax04.set_xlim(-10.0,370.0)

    # Turn on grids
    ax01.grid(True)
    ax02.grid(True)
    ax03.grid(True)
    ax04.grid(True)

    # set label names
    ax01.set_xlabel("Axial height (cm)")
    ax01.set_ylabel("Neutron flux (/s)")
    ax02.set_xlabel("Axial height (cm)")
    ax02.set_ylabel("Fuel temperature (K)")
    ax03.set_xlabel("Time (s)")
    ax03.set_ylabel("Reactivity")
    ax04.set_xlabel("Axial height (cm)")
    ax04.set_ylabel("Coolant temperature (K)")

    # set plots
    p1, = ax01.plot([],[],'k-')
    p2, = ax02.plot([],[],'r-')
    p3, = ax03.plot([],[],'g-')
    p4, = ax04.plot([],[],'b-')

    def init():
        p1.set_data([], [])
        p2.set_data([], [])
        p3.set_data([], [])
        p4.set_data([], [])
        return p1,p2,p3,p4

    def animate(k):
        p1.set_data(x,phi[k,:])
        p2.set_data(x,Tf[k,:])
        p3.set_data(t[:k+1],rho[:k+1])
        p4.set_data(x,Tw[k,:])
        return p1, p2, p3, p4

    # interval: draw new frame every 'interval' ms
    # frames: number of frames to draw
    simulation = animation.FuncAnimation(fig, animate, init_func=init,frames=K, blit=True)
    #writergif = animation.PillowWriter(fps=30)
    writervideo = animation.FFMpegWriter(fps=(K-1)/5)
    #simulation.save('td_sim_%i.gif'%(K-1), writer=writergif)
    simulation.save('IC_%i.mp4'%(K-1), writer=writervideo)
    plt.clf()

    plt.plot(t,power,'k')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W/cc)')
    plt.savefig('td_pow_%i.svg'%(K-1), dpi=1200, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()