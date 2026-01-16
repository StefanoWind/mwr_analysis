# -*- coding: utf-8 -*-
"""
Radio frequency interference removal
"""
import os
cd=os.path.dirname(__file__)
import warnings
from scipy import stats
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_hbb=os.path.join(cd,'data','Assignment 2 Dataset','propietary_data3.csv')

#qc
min_B=0#minimum values of PSD
max_B=1000#maximum values of PSD
min_freq=10#minimum frequency
max_freq=14000#maximum frequency
window_time=20#bins size in time for kurtosis
window_freq=100#bins size in frequency for kurtosis
min_kurtosis=-0.285#minimum kurtosis (De Roo et al., 2019)
max_kurtosis=0.279#maximum kurtosis (De Roo et al., 2019)
sigma_time=10#Gaussian smoothing extent in time
sigma_freq=50#Gaussian smoothing extent in frequency

#graphics
sel_clean=[50,6000]#clean sample
sel_rfi=[120,7500]#rfi-corrupted sample

#%% Initialiazation

#read PSD data
df=pd.read_csv(source_hbb, header=None)
B_raw=xr.DataArray(df.values,coords={'time':np.arange(len(df.index)),'freq':np.arange(len(df.columns))})

#bins for kurtosis cluster analysis
bins_time=np.arange(-0.5,len(B_raw.time)+window_time,window_time)
bins_freq=np.arange(-0.5,len(B_raw.freq)+window_freq,window_freq)
time_avg=(bins_time[:-1]+bins_time[1:])/2
freq_avg=(bins_freq[:-1]+bins_freq[1:])/2

#%% Function
def inpaint(da,sigma1=5,sigma2=5):
    '''
    Inpaint missing values with Guassian filter
    '''
    
    from scipy.ndimage import gaussian_filter

    #mask nans
    da0 = da.values.copy()
    mask = np.isnan(da0)
    da0[mask] = 0
    weights = (~mask).astype(float)
    
    #apply Gaussian filter
    da_smooth =      gaussian_filter(da0, sigma=(sigma1, sigma2))#mean
    weights_smooth = gaussian_filter(weights, sigma=(sigma1, sigma2))#normalization factor
    
    #intpaint
    da_inpaint=da0.copy()
    da_inpaint[mask] = (da_smooth / weights_smooth)[mask]
    
    return xr.DataArray(da_inpaint,coords=da.coords)

#%% Main

#preliminary qc (thresholding)
B=B_raw.where(B_raw>=min_B).where(B_raw<=max_B)

#bin-kurtosis
time_grid,freq_grid=np.meshgrid(B.time.values,B.freq.values,indexing='ij')
B_grid=B.values
real=~np.isnan(time_grid+freq_grid+B_grid)

kurt_avg=stats.binned_statistic_2d(time_grid[real],freq_grid[real],B_grid[real],
                                   statistic=lambda x: stats.kurtosis(x),bins=[bins_time,bins_freq])[0]

#RFI flag
kurt_avg_da=xr.DataArray(kurt_avg,coords={'time':time_avg,'freq':freq_avg})
kurt_int=kurt_avg_da.interp(time=B.time,freq=B.freq,method='nearest')
rfi_flag=(kurt_int<min_kurtosis)+(kurt_int>max_kurtosis)
rejection_rate=np.sum(rfi_flag)/np.size(rfi_flag)*100

#corrected radiance
B_qc=inpaint(B.where(rfi_flag==0),sigma_time,sigma_freq)

#final qc
B_qc=B_qc.where(B.freq>=min_freq).where(B.freq<=max_freq)

#%% Plots
plt.close('all')

#radiance
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2, 3,width_ratios=[0.25,1,0.05],height_ratios=[1,0.25])

ax=fig.add_subplot(gs[0,0])
plt.semilogx(B.mean(dim='time'),B.freq,'-k',label='Mean',linewidth=2)
plt.semilogx(B.std(dim='time'),B.freq,'--r',label='StDev')
plt.xlabel(r'$log_{10}(B)$')
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
plt.plot(B.time.values[sel_clean[0]],B.freq.values[sel_clean[1]],'xk')
plt.plot(B.time.values[sel_rfi[0]],B.freq.values[sel_rfi[1]],'xk')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid()
cax=fig.add_subplot(gs[0,2])
plt.colorbar(pc,cax=cax,label=r'$log_{10}(B)$')

ax=fig.add_subplot(gs[1,1])
plt.semilogy(B.time,B.sum(dim='freq'),'-k')
plt.xlabel('Time')
plt.ylabel('Total Power')
plt.xlim([B.time[0],B.time[-1]])
plt.grid()

#sample PDFs
plt.figure()
B_clean=B.where(B.time>=B.time.values[sel_clean[0]]-window_time/2,drop=True)\
         .where(B.time< B.time.values[sel_clean[0]]+window_time/2,drop=True)\
         .where(B.freq>=B.freq.values[sel_clean[1]]-window_freq/2,drop=True)\
         .where(B.freq< B.freq.values[sel_clean[1]]+window_freq/2,drop=True)
plt.hist(B_clean.values.ravel(),color='k',bins=50)
plt.grid()
plt.xlabel('$B$')
plt.ylabel('Counts')
plt.tight_layout()

plt.figure()
B_rfi=B.where(B.time>=B.time.values[sel_rfi[0]]-window_time/2,drop=True)\
         .where(B.time< B.time.values[sel_rfi[0]]+window_time/2,drop=True)\
         .where(B.freq>=B.freq.values[sel_rfi[1]]-window_freq/2,drop=True)\
         .where(B.freq< B.freq.values[sel_rfi[1]]+window_freq/2,drop=True)
plt.hist(B_rfi.values.ravel(),color='k',bins=50)
plt.grid()
plt.xlabel('$B$')
plt.ylabel('Counts')
plt.tight_layout()

#RFI flag
fig=plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(1, 4,width_ratios=[1,1,1,0.05])

ax=fig.add_subplot(gs[0])
pc=rfi_flag.plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=0,
    vmax=1,
    cmap="RdYlGn_r",
    add_colorbar=False
)
plt.xlabel('Time')
plt.ylabel(r'$\nu$')
plt.title(f'RFI flag ({rejection_rate:0.01f}% rejected)')

ax=fig.add_subplot(gs[1])
pc=kurt_avg_da.plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=-1,
    vmax=1,
    cmap="seismic",
    add_colorbar=False
)
ax.set_ylabel('')
ax.set_yticklabels([])
plt.xlabel('Time')
plt.title('Bin average')

ax=fig.add_subplot(gs[2])
pc=kurt_int.plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=-1,
    vmax=1,
    cmap="seismic",
    add_colorbar=False
)
ax.set_ylabel('')
ax.set_yticklabels([])
plt.xlabel('Time')
plt.title('Interpolated')
cax=fig.add_subplot(gs[3])
plt.colorbar(pc,cax=cax,label='Fisher kurtosis')

#corrected radiance
fig=plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(1, 4,width_ratios=[1,1,1,0.05])

ax=fig.add_subplot(gs[0])
pc=np.log10(B_raw).plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=np.log10(np.nanpercentile(B,5)),
    vmax=np.log10(np.nanpercentile(B,99)),
    cmap="hot",
    add_colorbar=False
)
plt.xlabel('Time')
plt.ylabel(r'$\nu$')
plt.title('Raw')

ax=fig.add_subplot(gs[1])
pc=np.log10(B.where(rfi_flag==0)).plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=np.log10(np.nanpercentile(B,5)),
    vmax=np.log10(np.nanpercentile(B,99)),
    cmap="hot",
    add_colorbar=False
)
ax.set_ylabel('')
ax.set_yticklabels([])
plt.xlabel('Time')
plt.title('RFI-free')

ax=fig.add_subplot(gs[2])
pc=np.log10(B_qc).plot(
    ax=ax,
    x='time',
    y='freq',
    vmin=np.log10(np.nanpercentile(B,5)),
    vmax=np.log10(np.nanpercentile(B,99)),
    cmap="hot",
    add_colorbar=False
)
ax.set_ylabel('')
ax.set_yticklabels([])
plt.xlabel('Time')
plt.title('RFI-corrected')
cax=fig.add_subplot(gs[3])
plt.colorbar(pc,cax=cax,label=r'$log_{10}(B)$')

