#pragma once
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gpu_helpers.h"

namespace imaging_core {
  namespace data {
    using namespace std;
    using namespace boost::python;
    
    template <typename cube_dtype> class cube_dtype_traits{};
    template <> class cube_dtype_traits<float>{
      public:
	static const size_t npy_dtype = NPY_COMPLEX64;
	typedef thrust::complex<float> c_dtype;
    };
    template <> class cube_dtype_traits<double>{
      public: 
	static const size_t npy_dtype = NPY_COMPLEX128;
	typedef thrust::complex<double> c_dtype;
    };
    /**
     * Wraps a regular uv voxel cube that has dimensions nband x ncorrelations x nv x nu
     * This allocates and manages device memory.
     */
    template <typename cube_dtype>
    class regular_uv_cube{
    private:
      bool m_resources_aquired;  
      //Raw host pointers: don't assume responsibility for these:
      typename cube_dtype_traits<cube_dtype>::c_dtype * m_p_regular_grid;
      //Device vectors:
      thrust::device_vector<typename cube_dtype_traits<cube_dtype>::c_dtype> m_d_regular_grid;
      //Attributes:
      size_t m_num_correlations;
      size_t m_num_bands;
      size_t m_nv;
      size_t m_nu;
    public:
      regular_uv_cube(numeric::array & regular_cube):m_resources_aquired(false){
	//Check the cube is a contiguous c-ordered block
	if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(regular_cube.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(regular_cube.ptr())->descr->type_num != (cube_dtype_traits<cube_dtype>::npy_dtype))
	    throw std::runtime_error("Regular cube must be contiguous complex c array");
	//Check dimensions of the image block
	size_t nd_cube = len(regular_cube.attr("shape"));
	  if ((nd_cube) != 4)
	    throw std::runtime_error("Regular cube must have dimensions [NBand x Ncorr x Nv x Nu]");
	//Grab pointers:
	m_p_regular_grid = reinterpret_cast<typename cube_dtype_traits<cube_dtype>::c_dtype *>(reinterpret_cast<PyArrayObject*>(regular_cube.ptr())->data);
	//Descriptors:
	size_t nbands = extract<size_t>(regular_cube.attr("shape")[0]);
	size_t ncorrs = extract<size_t>(regular_cube.attr("shape")[1]);
	size_t nv = extract<size_t>(regular_cube.attr("shape")[2]);
	size_t nu = extract<size_t>(regular_cube.attr("shape")[2]);
	m_num_bands = nbands;
	m_num_correlations = ncorrs;
	m_nv = nv;
	m_nu = nu;
	//Finally host and device allocate:
	checked_acquire_persistant();
	cudaSafeCall(cudaDeviceSynchronize());
      }
      virtual ~regular_uv_cube(){
	checked_release_persistant();
      }
      regular_uv_cube(const regular_uv_cube & lvalue) = delete;
      regular_uv_cube(regular_uv_cube && rvalue){
	*this = std::move(rvalue);
      }
      regular_uv_cube& operator=(const regular_uv_cube & lvalue) = delete;
      regular_uv_cube& operator=(regular_uv_cube && rvalue) {
	m_resources_aquired = rvalue.m_resources_aquired;
	rvalue.m_resources_aquired = false;
	
	this->m_p_regular_grid = rvalue.m_p_regular_grid;
	rvalue.m_p_regular_grid = nullptr;
	m_d_regular_grid = std::move(rvalue.m_d_regular_grid);
	cudaSafeCall(cudaDeviceSynchronize());
	return *this;
      }
      
      regular_uv_cube & checked_acquire_persistant(){ //For use in Python context management try catch finally release mechanism (or with mechanism)
	if (!m_resources_aquired){
	  //Acquire resources here:
	  size_t num_voxels = m_num_bands * m_num_correlations * m_nv *m_nu;
	  m_d_regular_grid = std::move(thrust::device_vector<typename cube_dtype_traits<cube_dtype>::c_dtype>(m_p_regular_grid, m_p_regular_grid + num_voxels));
	}
	m_resources_aquired = true;
	cudaSafeCall(cudaDeviceSynchronize());
	return *this;
      }
      void checked_release_persistant(boost::python::object except_type = boost::python::object(),
				      boost::python::object exception_value = boost::python::object(),
				      boost::python::object traceback = boost::python::object()){ //For use in Python context management try catch finally release mechanism
	if (m_resources_aquired){	  
	  //not responsible for the raw pointers passed in originally:
	  #define release_array(arr) {arr.clear(); arr.shrink_to_fit();}
	  release_array(m_d_regular_grid);
	  cudaSafeCall(cudaDeviceSynchronize());
	}
	m_resources_aquired = false;
      }
      void sync_device2host(){
	cudaSafeCall(cudaDeviceSynchronize());
	size_t num_voxels = m_num_bands * m_num_correlations * m_nv *m_nu;
	thrust::copy_n(m_d_regular_grid.begin(),num_voxels,m_p_regular_grid);
	cudaSafeCall(cudaDeviceSynchronize());
      }
      void sync_host2device(){
	cudaSafeCall(cudaDeviceSynchronize());
	size_t num_voxels = m_num_bands * m_num_correlations * m_nv *m_nu;
	thrust::copy_n(m_p_regular_grid,num_voxels,m_d_regular_grid.begin());
	cudaSafeCall(cudaDeviceSynchronize());
      }
    };
  }
}