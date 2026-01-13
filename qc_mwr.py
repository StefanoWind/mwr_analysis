# -*- coding: utf-8 -*-
"""
QC MWR data
"""

import os
cd=os.path.dirname(__file__)
import sys
import warnings
from datetime import datetime
import yaml
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

#%% Inputs
source_hbb=os.path.join(cd,'data','Assignment 1 Dataset','P_calibration_hot.csv')
source_cbb=os.path.join(cd,'data','Assignment 1 Dataset','P_calibration_cold.csv')
source_sky=os.path.join(cd,'data','Assignment 1 Dataset','P_scene.csv')

T_hbb=300
T_cbb=80

#%% Initialization
df_hbb=pd.read_csv(source_hbb, header=None)
df_cbb=pd.read_csv(source_cbb, header=None)
df_sky=pd.read_csv(source_sky, header=None)

Data=xr.Dataset()
Data['P_hbb']=xr.DataArray(df_hbb.values,coords={'time':np.arange(len(df_hbb.index)),'freq':np.arange(len(df_hbb.columns))})
Data['P_cbb']=xr.DataArray(df_cbb.values,coords={'time':np.arange(len(df_cbb.index)),'freq':np.arange(len(df_cbb.columns))})
Data['P_sky']=xr.DataArray(df_sky.values,coords={'time':np.arange(len(df_sky.index)),'freq':np.arange(len(df_sky.columns))})


#%% Main
Data['Y']=Data.P_hbb/Data.P_cbb
Data['T_sys']=(T_hbb-Data.Y*T_cbb)/(Data.Y-1)
Data['G']=(Data.P_hbb-Data.P_cbb)/(T_hbb-T_cbb)
Data['T_sky']=Data.P_sky/Data.G-Data.T_sys

#%% Plots
plt.close('all')
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


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.pcolor(Data.time,Data.freq,Data.T_sys.T,cmap='coolwarm')
plt.xlabel('Time')
plt.ylabel(r'$\nu$')
plt.colorbar(label=r'$T_{sys}$ [K]')
plt.subplot(1,2,2)
plt.pcolor(Data.time,Data.freq,Data.T_sky.T,cmap='coolwarm')
plt.xlabel('Time')
plt.colorbar(label=r'$T_{sky}$ [K]')

