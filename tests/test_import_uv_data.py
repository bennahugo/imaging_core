#!/usr/bin/python
import unittest
from imaging_core.cbuild.resamplers.bda_resampler import bda_resampler_grid_single_correlation
from imaging_core.cbuild.data.irregular_uv_data import irregular_uv_data
import numpy as np

class class_test_uv_data_wrapper(unittest.TestCase):
    ncorrs = 4
    nchans = 64
    nrows = 1000
    
    uvw = None
    vis = None
    flags = None
    imgweights = None
    ant1ids = None
    ant2ids = None
    time = None
    ref_freqs = None
    uvw_copy = None
    vis_copy = None
    flags_copy = None
    imgweights_copy = None
    ant1ids_copy = None
    ant2ids_copy = None
    time_copy = None
    ref_freqs_copy = None
    
    #generates samples from a random distribution:
    def gen_samples_host(self):
      if self.uvw is not None:
	self.uvw[...] = np.random.rand(self.nrows,3).astype(np.float32)
      else:
	self.uvw  = np.random.rand(self.nrows,3).astype(np.float32)
      if self.vis is not None:
	self.vis[...] = np.random.rand(self.nrows,self.nchans,self.ncorrs).astype(np.complex64)
      else:
	self.vis = np.random.rand(self.nrows,self.nchans,self.ncorrs).astype(np.complex64)
      if self.flags is not None:
	self.flags[...] = np.round(np.random.rand(self.nrows,self.nchans)).astype(np.bool)
      else:
	self.flags = np.round(np.random.rand(self.nrows,self.nchans)).astype(np.bool)
      if self.imgweights is not None:
	self.imgweights[...] = np.random.rand(self.nrows,self.nchans).astype(np.float32)
      else:
	self.imgweights = np.random.rand(self.nrows,self.nchans).astype(np.float32)
      if self.ant1ids is not None:
	self.ant1ids[...] = np.random.randint(64,size=[self.nrows]).astype(np.uintp)
      else:
	self.ant1ids = np.random.randint(64,size=[self.nrows]).astype(np.uintp)
      if self.ant2ids is not None:
	self.ant2ids[...] = np.random.randint(64,size=[self.nrows]).astype(np.uintp)
      else:
	self.ant2ids = np.random.randint(64,size=[self.nrows]).astype(np.uintp)
      if self.time is not None:
	self.time[...] = np.random.rand(self.nrows).astype(np.float32)
      else:
	self.time = np.random.rand(self.nrows).astype(np.float32)
      if self.ref_freqs is not None:
	self.ref_freqs[...] = np.random.rand(self.nchans).astype(np.float32)
      else:
	self.ref_freqs = np.random.rand(self.nchans).astype(np.float32)
    
    #deep copies to check integrity against:
    def deep_copy_host(self):  
      self.uvw_copy = self.uvw.copy()
      self.vis_copy = self.vis.copy()
      self.flags_copy = self.flags.copy()
      self.imgweights_copy = self.imgweights.copy()
      self.ant1ids_copy = self.ant1ids.copy()
      self.ant2ids_copy = self.ant2ids.copy()
      self.time_copy = self.time.copy()
      self.ref_freqs_copy = self.ref_freqs.copy()
      
    #resets the host memory to zeros (this will change the host pointers in the data wrapper):
    def clean_host_memory(self):
      self.uvw.fill(0)
      self.vis.fill(0)
      self.flags.fill(False)
      self.imgweights.fill(0)
      self.ant1ids.fill(0)
      self.ant2ids.fill(0)
      self.time.fill(0)
      self.ref_freqs.fill(0)
      
    #check the host copies for equality to the deep copies on the host:
    def check_integrity(self):	
      assert np.allclose(self.uvw,self.uvw_copy)
      assert np.allclose(self.vis,self.vis_copy)
      assert np.allclose(self.flags,self.flags_copy)
      assert np.allclose(self.imgweights,self.imgweights_copy)
      assert np.allclose(self.ant1ids,self.ant1ids_copy)
      assert np.allclose(self.ant2ids,self.ant2ids_copy)
      assert np.allclose(self.time,self.time_copy)
      assert np.allclose(self.ref_freqs,self.ref_freqs_copy)
      
    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def test_construct_and_transfer(self):  
      #construct uv_data thereby transferring the samples host to device:
      self.gen_samples_host()
      self.deep_copy_host()
      with irregular_uv_data(self.uvw,
			     self.vis,
			     self.flags,
			     self.imgweights,
			     self.ant1ids,
			     self.ant2ids,
			     self.time,
			     self.ref_freqs) as samples:
	#basic copy back
	self.clean_host_memory()
	samples.sync_device2host()
	self.check_integrity()
	
    def test_update_host(self):  
      #construct uv_data thereby transferring the original samples host to device:
      self.gen_samples_host()
      self.deep_copy_host()
      with irregular_uv_data(self.uvw,
			     self.vis,
			     self.flags,
			     self.imgweights,
			     self.ant1ids,
			     self.ant2ids,
			     self.time,
			     self.ref_freqs) as samples:
	#clean host and regenerate samples host side
	self.clean_host_memory()
	self.gen_samples_host()
	self.deep_copy_host()
	samples.sync_host2device()
	self.clean_host_memory()
	samples.sync_device2host()
	self.check_integrity()
	
    def test_not_contiguous_arrays(self):
      self.gen_samples_host()
      #just test one for now to make sure
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(np.zeros([self.nrows*3,3], dtype=np.float32)[0:self.nrows*3:3,:],
					self.vis,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      #just test one array with fortran ordering
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(np.zeros([3,self.nrows], dtype=np.float32).T,
					self.vis,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      #check correct types:
      foobar = np.zeros([100],dtype=object)
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(foobar,self.vis,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,foobar,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,foobar,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,self.flags,foobar,
				self.ant1ids,self.ant2ids,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,self.flags,self.imgweights,
				foobar,self.ant2ids,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,self.flags,self.imgweights,
				self.ant1ids,foobar,self.time,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,foobar,self.ref_freqs))
      self.assertRaises(RuntimeError, 
			lambda :irregular_uv_data(self.uvw,self.vis,self.flags,self.imgweights,
				self.ant1ids,self.ant2ids,self.time,foobar))
      #if the shapes passed in was incorrect we should see it in the copied data in the least
      
if __name__ == "__main__":
    unittest.main()