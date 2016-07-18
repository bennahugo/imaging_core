#!/usr/bin/python

from imaging_core.cbuild.resamplers.bda_resampler import bda_resampler_a_policy, bda_resampler_b_policy
from imaging_core.cbuild.resamplers.uv_data import uv_data
import numpy as np
uvw = np.zeros([1000,3],dtype=np.float64)
uvw[0,:] = [1,2,3]
samples = uv_data(uvw);
try:
  resampler = bda_resampler_a_policy(1024,1024,10,4,1000,64,4)
  resampler.nonregular2regular(samples)
finally:
  resampler.checked_release_persistant();
  
with bda_resampler_b_policy(1024,1024,10,4,1000,64,4) as resampler:
  resampler.nonregular2regular(samples)