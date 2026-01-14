import xarray as xr
import numpy as np

# Example DataArray
freq = np.linspace(20, 100, 10)
time = np.arange(50)
data = np.random.rand(10,50)
da = xr.DataArray(data, dims=["freq","time"], coords={"freq":freq, "time":time})

# Define bins along freq
bins = np.linspace(20, 100, 5)  # 4 bins

# Group by bins
grouped = da.groupby_bins("freq", bins)

# Apply a function (e.g., mean over each bin)
bin_means = grouped.mean(dim="freq")
