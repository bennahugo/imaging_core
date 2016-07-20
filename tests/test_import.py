#!/usr/bin/python
from imaging_core.cbuild.resamplers.bda_resampler import bda_resampler_grid_single_correlation
from imaging_core.cbuild.resamplers.uv_data import uv_data
import numpy as np
ncorrs = 4
nchans = 64
nrows = 1000
uvw = np.zeros([nrows,3],dtype=np.float32)
vis = np.zeros([nrows,nchans,ncorrs],dtype=np.complex64)
flags = np.zeros([nrows,nchans,ncorrs],dtype=np.bool)
imgweights = np.zeros([nrows,nchans],dtype=np.float32)
ant1ids = np.zeros([nrows],dtype=np.uintp)
ant2ids = np.zeros([nrows],dtype=np.uintp)
time = np.zeros([nrows],dtype=np.float32)
ref_freqs = np.zeros([nchans],dtype=np.float32)

vis[1,5,:] = [5,6,7,8]
uvw[0,:] = [1,2,3]
samples = uv_data(uvw,
		  vis,
		  flags,
		  imgweights,
		  ant1ids,
		  ant2ids,
		  time,
		  ref_freqs);
resampler = None
try:
  resampler = bda_resampler_grid_single_correlation()
  resampler.nonregular2regular(samples)
finally:
  resampler.checked_release_persistant();
  
with bda_resampler_grid_single_correlation() as resampler:
  resampler.nonregular2regular(samples,True,True)