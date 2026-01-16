# -*- coding: utf-8 -*-
"""
Planck vs Rayleigh-Jeans check
"""
import numpy as np
from scipy.constants import h, k, c
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi']=500

#%% Inputs
nu=np.arange(0,300)*1e9#[Hz] frequency
T=3#[K] black body temperature

#%% Main
B=2*h*nu**3/c**2/(np.exp(h*nu/(k*T))-1)#Planck curve
B_rj=2*k*nu**2/c**2*T#Rayleigh-Jeans approximation

#%% Plots
plt.figure()
plt.plot(nu/10**9,B,'k',label='Plank')
plt.plot(nu/10**9,B_rj,'--r',label='Rayleigh-Jeans')
plt.grid()
plt.legend()
plt.title(r'$T='+str(T)+'$ K')
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$B$ [W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$ ]')
