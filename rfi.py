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
from scipy import stats
import functools
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

#%% Inputs
source_hbb=os.path.join(cd,'data','Assignment 2 Dataset','sermon_c414_20250609T114551_20250609T114719_level1b.csv')
window=20

#%% Initialiazation
df=pd.read_csv(source_hbb, header=None)

B=xr.DataArray(df.values,coords={'time':np.arange(len(df.index)),'freq':np.arange(len(df.columns))})
B=B.where(B>0)

# B_avg=B.rolling(time=window,center=True).mean()
# dB=B-B_avg
# m4=(dB**4).rolling(time=window,center=True).mean()
# m2=(dB**2).rolling(time=window,center=True).mean()
# kurt=m4/m2**2

bins=np.arange(0,len(B.time),20)

# df["bins"] = pd.cut(df.x, bins(df.x, config.dx))

# groups = df.groupby(
#     ["xbins", "ybins", "zbins", "timebins"], group_keys=False, observed=True
# )


#%% Plots
plt.close('all')
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2, 3,width_ratios=[0.25,1,0.05],height_ratios=[1,0.25])
ax=fig.add_subplot(gs[0,0])
plt.semilogx(B.mean(dim='time'),B.freq,'-k',label='Mean',linewidth=2)
plt.semilogx(B.std(dim='time'),B.freq,'--r',label='StDev')
plt.xlabel('PSD')
plt.ylabel(r'$\nu$')
plt.ylim([B.freq[0],B.freq[-1]])
plt.legend()
plt.grid()

ax=fig.add_subplot(gs[0,1])
pc=np.log10(B).plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=np.log10(np.nanpercentile(B,5)),
    vmax=np.log10(np.nanpercentile(B,99)),
    cmap="hot",
    add_colorbar=False
)
ax.set_xlabel('')
ax.set_ylabel('')
# ax.set_xticklabels([])
# ax.set_yticklabels([])
plt.grid()
cax=fig.add_subplot(gs[0,2])
plt.colorbar(pc,cax=cax,label='PSD')

ax=fig.add_subplot(gs[1,1])
plt.semilogy(B.time,B.sum(dim='freq'),'-k')
plt.xlabel('Time')
plt.ylabel('Total Power')
plt.xlim([B.time[0],B.time[-1]])
plt.grid()

fig=plt.figure()
ax=fig.add_subplot(111)
pc=kurt.plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=2,
    vmax=4,
    cmap="hot",
)
plt.xlabel('Time')
plt.ylabel(r'$\nu$')
plt.grid()