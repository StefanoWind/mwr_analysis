# -*- coding: utf-8 -*-
"""
Dummy plots for dashboard
"""
import os
cd=os.path.dirname(__file__)
import warnings
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Functions
def ar(mean,sigma,ac,N):
    '''
    AR(1) process
    '''
    y=np.zeros(N)
    for i in range(1,N):
        y[i]=y[i-1]*ac+np.random.normal(0,sigma)
    y+=mean
    return y

#%% Plots

#thermal stability
fig=plt.figure(figsize=(18,6))
ax=fig.add_subplot(3,1,1)
plt.plot(np.arange(1000),ar(40,1,0.9,1000)+np.sin(6.28/1000*10*np.arange(1000))*5,'k')
plt.ylabel(r'$T$ payload')
plt.grid()
ax.set_xticklabels([])
ax=fig.add_subplot(3,1,2)
plt.plot(np.arange(1000),ar(300,1,0.1,1000),'r')
plt.ylabel(r'$T$ ND')
plt.grid()
ax.set_xticklabels([])
ax=fig.add_subplot(3,1,3)
plt.plot(np.arange(1000),ar(80,1,0.1,1000),'g')
plt.ylabel(r'$T$ cold')
plt.xlabel('Time')
plt.grid()
ax.set_xticklabels([])

#MWR calibration
fig=plt.figure(figsize=(18,6))
ax=fig.add_subplot(3,1,1)
plt.plot(np.arange(1000),ar(250,5,0.9,1000)+np.sin(6.28/1000*10*np.arange(1000))*2,'r',label='Opaque channel')
plt.plot(np.arange(1000),ar(100,10,0.9,1000)+np.sin(6.28/1000*10*np.arange(1000))*5,'b',label='Transparent channel')
plt.ylabel(r'$T_b$')
plt.grid()
ax.set_xticklabels([])
plt.legend()
ax=fig.add_subplot(3,1,2)
plt.plot(np.arange(1000),ar(1,0.1,0.5,1000)+np.abs(np.arange(1000)<300),'k')
plt.ylabel(r'Gain')
plt.grid()
ax.set_xticklabels([])
ax=fig.add_subplot(3,1,3)
plt.plot(np.arange(1000),ar(0.1,1,0.5,1000)-np.abs(np.arange(1000)<300)*5,'k')
plt.ylabel(r'NDET')
plt.xlabel('Time')
plt.grid()

#PDFs of Tb
fig=plt.figure()
plt.hist(np.random.normal(3000,40,10000),bins=21,color='b',label='Orbit ensemble',alpha=0.5,density=True)
plt.hist(np.random.lognormal(3,1.2,200)+2900,bins=21,color='r',label='last scan',alpha=0.5,density=True)
plt.xlabel('Counts')
plt.grid()
plt.legend()

fig=plt.figure()
plt.hist(np.random.normal(300,40,10000),bins=21,color='b',label='Orbit ensemble',alpha=0.5,density=True)
plt.hist(np.random.lognormal(3,1.2,100)+200,bins=21,color='r',label='last scan',alpha=0.5,density=True)
plt.xlabel('$T_b$ at 100 GHz')
plt.grid()
plt.legend()

#Spectrum
fig=plt.figure(figsize=(18,6))
ax=fig.add_subplot(3,1,1)
nu=np.arange(1000)/1000
plt.plot(nu,(nu**2-np.sin(6.28*nu)*nu+np.random.normal(0,.1,1000)+(np.abs(nu-0.4)<0.1)*(nu/0.01==np.round(nu/0.01)))*100,'k')
plt.ylabel('Last $T_b$')
ax.set_xticklabels([])
plt.grid()

ax=fig.add_subplot(3,1,2)
nu=np.arange(1000)/1000
plt.fill_between(nu,np.random.normal(80,10,1000),color='g',alpha=0.5)
plt.ylabel('Data availability')
plt.xlabel('Frequency')
plt.grid()

#Retrieval
fig=plt.figure(figsize=(10,18))
z=np.arange(100)/10
ax=fig.add_subplot(1,2,1)
plt.plot(np.sin(z/10*6.28)**2*50*z/10+100+np.random.normal(0,10,100),z,'r')
plt.ylabel('Altitude')
plt.xlabel('Last temperature profile')
plt.grid()
ax=fig.add_subplot(1,2,2)
plt.plot(np.cos(z/20*6.28)**2*5+10-z/5+np.random.normal(0,1,100),z,'b')
ax.set_yticklabels([])
plt.xlabel('Last WVMR profile')
plt.grid()

fig=plt.figure()
ax=fig.add_subplot(2,1,1)
plt.bar(['HRRR','ERA-5','GEOS','ECMWF'],[-0.5,1,0.3,-0.1],color='b',width=0.5)
plt.ylabel('Monthly bias')
ax.set_xticklabels([])
plt.grid()
ax=fig.add_subplot(2,1,2)
plt.bar(['HRRR','ERA-5','GEOS','ECMWF'],[2,1.4,3,4],color='r',width=0.5)
plt.ylabel('Monthly RMSE')
plt.grid()