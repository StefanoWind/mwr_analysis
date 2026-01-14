# -*- coding: utf-8 -*-
"""
QC MWR data
"""

import os
cd=os.path.dirname(__file__)
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_hbb=os.path.join(cd,'data','Assignment 1 Dataset','P_calibration_hot.csv')#hot BB source
source_cbb=os.path.join(cd,'data','Assignment 1 Dataset','P_calibration_cold.csv')#cold BB source
source_sky=os.path.join(cd,'data','Assignment 1 Dataset','P_scene.csv')#scene source

T_hbb=300#[K] HBB temperature
T_cbb=80#[K] CBB temperature

#%% Initialization

#read data 
df_hbb=pd.read_csv(source_hbb, header=None)
df_cbb=pd.read_csv(source_cbb, header=None)
df_sky=pd.read_csv(source_sky, header=None)

#convert to xarray
Data=xr.Dataset()
Data['P_hbb']=xr.DataArray(df_hbb.values,
                           coords={'time':np.arange(len(df_hbb.index)),
                                   'freq':np.arange(len(df_hbb.columns))})
Data['P_cbb']=xr.DataArray(df_cbb.values,
                           coords={'time':np.arange(len(df_cbb.index)),
                                   'freq':np.arange(len(df_cbb.columns))})
Data['P_sky']=xr.DataArray(df_sky.values,
                           coords={'time':np.arange(len(df_sky.index)),
                                   'freq':np.arange(len(df_sky.columns))})

#%% Main
Data['Y']=Data.P_hbb/Data.P_cbb#Y-factor
Data['T_sys']=(T_hbb-T_cbb*Data.Y)/(Data.Y-1)#noise temperature
Data['G']=(Data.P_hbb-Data.P_cbb)/(T_hbb-T_cbb)#gain
Data['T_sky']=Data.P_sky/Data.G-Data.T_sys#scene temperature

#check with direct linear extrapolation
Data['T_sky_check']=(T_hbb-T_cbb)/(Data.P_hbb-Data.P_cbb)*(Data.P_sky-Data.P_cbb)+T_cbb
print(f'Max difference in calibration formula check {np.nanmax(Data.T_sky-Data.T_sky_check):0.02f} K')

#variations of BB radiances
Data['DP_hbb']=(Data.P_hbb-Data.P_hbb.mean(dim='time'))/Data.P_hbb*100
Data['DP_cbb']=(Data.P_cbb-Data.P_cbb.mean(dim='time'))/Data.P_cbb*100
Data['DP_sky']=(Data.P_sky-Data.P_sky.mean(dim='time'))/Data.P_sky*100

#correlation of BB radiances
Data['corr_hbb'] =xr.DataArray(np.corrcoef(Data.DP_hbb.values.T),
                               coords={'freq1':np.arange(len(df_hbb.columns)),
                                       'freq2':np.arange(len(df_hbb.columns))})
Data['corr_cbb'] =xr.DataArray(np.corrcoef(Data.DP_cbb.values.T),
                               coords={'freq1':np.arange(len(df_cbb.columns)),
                                       'freq2':np.arange(len(df_cbb.columns))})
Data['corr_sky'] =xr.DataArray(np.corrcoef(Data.DP_sky.values.T),
                               coords={'freq1':np.arange(len(df_sky.columns)),
                                       'freq2':np.arange(len(df_sky.columns))})

#%% Plots
plt.close('all')

#all radiances
plt.figure(figsize=(18,4))
plt.subplot(1,3,1)
pc=plt.pcolor(Data.time,Data.freq,Data.P_hbb.T,cmap='plasma')
plt.ylabel(r'$\nu$')
plt.xlabel('Time')
plt.colorbar(label=r'$P_{HBB}$ [counts]')
plt.subplot(1,3,2)
plt.pcolor(Data.time,Data.freq,Data.P_cbb.T,cmap='plasma')
plt.xlabel('Time')
plt.colorbar(label=r'$P_{CBB}$ [counts]')
plt.subplot(1,3,3)
plt.pcolor(Data.time,Data.freq,Data.P_sky.T,cmap='plasma')
plt.colorbar(label=r'$P_{sky}$ [counts]')
plt.xlabel('Time')
plt.tight_layout()

#BB radiance stability
plt.figure(figsize=(18,4))
plt.subplot(1,3,1)
pc=plt.pcolor(Data.time,Data.freq,Data.DP_hbb.T,cmap='seismic',vmin=-1,vmax=1)
plt.ylabel(r'$\nu$')
plt.xlabel('Time')
plt.colorbar(label=r'$\Delta P_{HBB}$ [%]')
plt.subplot(1,3,2)
plt.pcolor(Data.time,Data.freq,Data.DP_cbb.T,cmap='seismic',vmin=-1,vmax=1)
plt.xlabel('Time')
plt.colorbar(label=r'$\Delta P_{CBB}$ [%]')
plt.subplot(1,3,3)
plt.pcolor(Data.time,Data.freq,Data.DP_sky.T,cmap='seismic',vmin=-1,vmax=1)
plt.xlabel('Time')
plt.colorbar(label=r'$\Delta P_{sky}$ [%]')
plt.tight_layout()

#channel-to-channel of radiance
plt.figure(figsize=(18,4))
plt.subplot(1,3,1)
pc=plt.pcolor(Data.freq1,Data.freq2,Data.corr_hbb.T,cmap='seismic',vmin=-1,vmax=1)
plt.ylabel(r'$\nu$')
plt.xlabel(r'$\nu$')
plt.colorbar(label=r'$\rho(\Delta P_{HBB})$')
plt.subplot(1,3,2)
pc=plt.pcolor(Data.freq1,Data.freq2,Data.corr_cbb.T,cmap='seismic',vmin=-1,vmax=1)
plt.xlabel(r'$\nu$')
plt.colorbar(label=r'$\rho(\Delta P_{CBB})$')
plt.subplot(1,3,3)
pc=plt.pcolor(Data.freq1,Data.freq2,Data.corr_sky.T,cmap='seismic',vmin=-1,vmax=1)
plt.xlabel(r'$\nu$')
plt.colorbar(label=r'$\rho(\Delta P_{sky})$')
plt.tight_layout()

#all temperatures
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.pcolor(Data.time,Data.freq,Data.T_sys.T,cmap='coolwarm')
plt.xlabel('Time')
plt.ylabel(r'$\nu$')
plt.colorbar(label=r'$T_{b,sys}$ [K]')
plt.subplot(1,2,2)
plt.pcolor(Data.time,Data.freq,Data.T_sky.T,cmap='coolwarm')
plt.xlabel('Time')
plt.colorbar(label=r'$T_{b,sky}$ [K]')
plt.tight_layout()
