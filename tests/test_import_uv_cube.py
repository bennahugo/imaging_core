#!/usr/bin/python
import unittest
from imaging_core.cbuild.data.regular_uv_cube import regular_uv_cube
import numpy as np

class class_test_uv_data_wrapper(unittest.TestCase):
  nbands = 10
  ncorrs = 4
  nv = 256
  nu = 256
  
  def test_construct_and_transfer(self):
    #generate some random cube
    cube = np.random.rand(self.nbands,self.ncorrs,self.nv,self.nu).astype(np.complex64)
    cube_copy = cube.copy()
    #transfer to device:
    with regular_uv_cube(cube) as uv_reg:
      #zero out host, transfer back and check integrity
      cube.fill(0)
      uv_reg.sync_device2host()
      assert np.allclose(cube,cube_copy)
      
  def test_host2device_sync(self):
    #generate some random cube
    cube = np.random.rand(self.nbands,self.ncorrs,self.nv,self.nu).astype(np.complex64)
    cube_copy = cube.copy()
    #transfer to device:
    with regular_uv_cube(cube) as uv_reg:
      #zero out host and set up new data
      cube.fill(0)
      cube[...] = np.random.rand(self.nbands,self.ncorrs,self.nv,self.nu).astype(np.complex64)
      cube_copy = cube.copy()
      #sync, reset host and transfer back
      uv_reg.sync_host2device()
      cube.fill(0)
      uv_reg.sync_device2host()
      assert np.allclose(cube,cube_copy)
      
  def test_exceptions(self):
     #check type and ordering:
     cube = np.empty([self.nbands,self.ncorrs,self.nv,self.nu],dtype=object)
     self.assertRaises(RuntimeError, lambda : regular_uv_cube(cube))
     cube = np.zeros([self.nu,self.nv,self.nv,self.ncorrs,self.nbands],dtype=np.complex64).T
     self.assertRaises(RuntimeError, lambda : regular_uv_cube(cube))
     cube = np.zeros([self.nbands*3,self.ncorrs,self.nv,self.nu],dtype=np.complex64)[0:self.nbands*3:3,:,:,:]
     self.assertRaises(RuntimeError, lambda : regular_uv_cube(cube))
if __name__ == "__main__":
    unittest.main()