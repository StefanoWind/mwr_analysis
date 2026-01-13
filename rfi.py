# -*- coding: utf-8 -*-
"""
Radio frequency interference removal
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
source_hbb=os.path.join(cd,'data','Assignment 2 Dataset','sermon_c414_20250609T114551_20250609T114719_level1b.csv')

#%% Fnctions
def symlog(x):
    
    return np.sign(x)*np.log(1+np.abs(x))

#%% Initialiazation
df=pd.read_csv(source_hbb, header=None)

T=xr.DataArray(df.values,coords={'time':np.arange(len(df.index)),'freq':np.arange(len(df.columns))})

#%% Plots
plt.close('all')
plt.figure(figsize=(18,4))
ax=plt.gca()
symlog(T).plot(
    ax=ax,
    x="time",
    y='freq',
    vmin=symlog(np.nanpercentile(T,1)),
    vmax=symlog(np.nanpercentile(T,99)),
    cmap="coolwarm",
    cbar_kwargs={'label': r"$T_b$ [K]"}
)


plt.figure()
