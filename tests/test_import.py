#!/usr/bin/python

import imaging_core.cbuild.resamplers.gridder as gridder
import imaging_core.cbuild.resamplers.uv_data as uv_data
import numpy as np
uvw = np.zeros([1000,3],dtype=np.float64)
uvw[0,:] = [1,2,3]
samples = uv_data.uv_data(uvw);
gridder.grid(samples)