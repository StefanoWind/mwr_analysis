# -*- coding: utf-8 -*-
"""
Planck and Rayleigh-Jeans
"""
import numpy as np
from scipy.constants import h, k, c
from matplotlib import pyplot as plt
#%% Inputs
nu=np.arange(0,300)*1e9

T=200

#%% Main
B=2*h*nu**3/c**2/(np.exp(h*nu/(k*T))-1)

B_rj=2*k*nu**2/c**2*T

#%% Plots
plt.figure()
plt.plot(nu,B,'k')
plt.plot(nu,B_rj,'--k')
plt.grid()
