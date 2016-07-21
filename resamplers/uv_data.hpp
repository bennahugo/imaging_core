#pragma once
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "gpu_helpers.h"

namespace imaging_core{
  namespace resamplers{
    using namespace std;
    using namespace boost::python;
    
    struct uvwT {
      float m_u,m_v,m_w;
      __device__ __host__ uvwT(float u=0, float v=0, float w=0): m_u(u),m_v(v),m_w(w){}
    };
    
    /**
     * Interferometer Measurement Data (uv_data) wrapper 
     * Wrapps host numpy arrays and is responsible for transfers to and from device
     */
    
    class uv_data
    {   
    private:
      bool m_resources_aquired;
      
      //Raw weak host pointers: do not assume responsibility for these:
      uvwT * m_p_uvw;
      thrust::complex<float> * m_p_vis;
      bool * m_p_flags;
      float * m_p_img_weights;
      size_t * m_p_ant_1_ids;
      size_t * m_p_ant_2_ids;
      float * m_p_measurement_time;
      float * m_p_channel_ref_freqs;
      //Device vectors:
      thrust::device_vector<uvwT> m_d_uvw;
      thrust::device_vector<thrust::complex<float>> m_d_vis;
      thrust::device_vector<bool> m_d_flags;
      thrust::device_vector<float> m_d_img_weights;
      thrust::device_vector<size_t> m_d_ant_1_ids;
      thrust::device_vector<size_t> m_d_ant_2_ids;
      thrust::device_vector<float> m_d_measurement_time;
      thrust::device_vector<float> m_d_channel_ref_freqs;
      //Attributes:
      size_t m_num_rows;
      size_t m_num_chan;
      size_t m_num_corrs;
    public:
      /**
       * Expects nrows number of visibility measurements where there are usually nbaselines x ntime steps worth of data
       * Multiple bands and fields must be stripped out of the data prior to construction of the wrapper
       * This wrapper does not deep copy the host data (ie. it is not a container). The calling python code
       * is responsible for managing the host data. As all persistant resources the API user is responsible
       * for releasing resources in a try...catch...finally block or using a with context manager. In a C++
       * this class is RAII complient and will release device resources upon destruction.
       */
      uv_data(numeric::array & uvw,
	      numeric::array & vis,
	      numeric::array & flags,
	      numeric::array & weights,
	      numeric::array & ant_1_ids,
	      numeric::array & ant_2_ids,
	      numeric::array & measurement_time,
	      numeric::array & channel_ref_freqs):m_resources_aquired(false){
	//Check that the arrays are contiguous c arrays
	{
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(uvw.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(uvw.ptr())->descr->type_num != NPY_FLOAT32)
	    throw std::runtime_error("UVW array must be contiguous float32 c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(vis.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(vis.ptr())->descr->type_num != NPY_COMPLEX64)
	    throw std::runtime_error("VIS array must be contiguous complex<float> c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(flags.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(flags.ptr())->descr->type_num != NPY_BOOL)
	    throw std::runtime_error("FLAGS array must be contiguous bool c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(weights.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(weights.ptr())->descr->type_num != NPY_FLOAT32)
	    throw std::runtime_error("WEIGHTS array must be contiguous float32 c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(ant_1_ids.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(ant_1_ids.ptr())->descr->type_num != NPY_UINTP)
	    throw std::runtime_error("ANT_1_IDS array must be contiguous size_t (uintp) c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(ant_2_ids.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(ant_2_ids.ptr())->descr->type_num != NPY_UINTP)
	    throw std::runtime_error("ANT_2_IDS array must be contiguous size_t (uintp) c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(measurement_time.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(measurement_time.ptr())->descr->type_num != NPY_FLOAT32)
	    throw std::runtime_error("Measurement_time array must be contiguous float32 c array");
	  if (!PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(channel_ref_freqs.ptr())) ||
	      reinterpret_cast<PyArrayObject*>(channel_ref_freqs.ptr())->descr->type_num != NPY_FLOAT32)
	    throw std::runtime_error("Channel reference frequency array must be contiguous float32 c array");
	}
	//Check dimensions of the arrays
	{
	  size_t nd_uvw = len(uvw.attr("shape"));
	  if ((nd_uvw) != 2 && extract<size_t>(uvw.attr("shape")[1] == 3))
	    throw std::runtime_error("UVW array must have dimensions [row x 3]");
	  size_t nd_vis = len(vis.attr("shape"));
	  if (nd_vis != 3)
	    throw std::runtime_error("VIS array must have dimensions [row x channel x correlation]");
	  size_t nd_flags = len(flags.attr("shape"));
	  if (nd_flags != 2)
	    throw std::runtime_error("FLAGS array must have dimensions [row x channel]");
	  size_t nd_weights = len(weights.attr("shape"));
	  if (nd_weights != 2)
	    throw std::runtime_error("IMAGE_WEIGHTS array must have dimensions [row x channel]");
	  size_t nd_ant_1_ids = len(ant_1_ids.attr("shape"));
	  if (nd_ant_1_ids != 1)
	    throw std::runtime_error("ANT_1_IDS array must have dimension [row]");
	  size_t nd_ant_2_ids = len(ant_2_ids.attr("shape"));
	  if (nd_ant_2_ids != 1)
	    throw std::runtime_error("ANT_2_IDS array must have dimension [row]");
	  size_t nd_measurement_time = len(measurement_time.attr("shape"));
	  if (nd_measurement_time != 1)
	    throw std::runtime_error("Measurement time array must have dimension [row]");
	  size_t nd_channel_ref_freqs = len(channel_ref_freqs.attr("shape"));
	  if (nd_channel_ref_freqs != 1)
	    throw std::runtime_error("Channel reference frequencies array must have dimension [channel]");
	  
	  if (extract<size_t>(uvw.attr("shape")[0]) != extract<size_t>(vis.attr("shape")[0]))
	    throw std::runtime_error("Number of visibilities do not correspond to number of visibility coordinates");
	  if (extract<size_t>(vis.attr("shape")[0]) != extract<size_t>(ant_1_ids.attr("shape")[0]))
	    throw std::runtime_error("Number of visibilities do not correspond to number of antenna 1 descriptors");
	  if (extract<size_t>(vis.attr("shape")[0]) != extract<size_t>(ant_2_ids.attr("shape")[0]))
	    throw std::runtime_error("Number of visibilities do not correspond to number of antenna 2 descriptors");
	  if (extract<size_t>(vis.attr("shape")[0]) != extract<size_t>(measurement_time.attr("shape")[0]))
	    throw std::runtime_error("Number of visibilities do not correspond to number of measurement time descriptors");
	  
	  if (extract<size_t>(vis.attr("shape")[1]) != extract<size_t>(flags.attr("shape")[1]))
	    throw std::runtime_error("Number of channels in visibilities and flags do not correspond");
	  if (extract<size_t>(vis.attr("shape")[1]) != extract<size_t>(weights.attr("shape")[1]))
	    throw std::runtime_error("Number of channels in visibilities and weights do not correspond");
	  if (extract<size_t>(vis.attr("shape")[1]) != extract<size_t>(channel_ref_freqs.attr("shape")[0]))
	    throw std::runtime_error("Number of channels in visibilities and reference frequencies do not correspond"); 
	}  
	//grab pointers
	{
	  m_p_uvw = reinterpret_cast<uvwT *>(reinterpret_cast<PyArrayObject*>(uvw.ptr())->data);
	  m_p_vis = reinterpret_cast<thrust::complex<float> *>(reinterpret_cast<PyArrayObject*>(vis.ptr())->data);
	  m_p_flags = reinterpret_cast<bool*>(reinterpret_cast<PyArrayObject*>(flags.ptr())->data);
	  m_p_img_weights = reinterpret_cast<float*>(reinterpret_cast<PyArrayObject*>(weights.ptr())->data);
	  m_p_ant_1_ids = reinterpret_cast<size_t*>(reinterpret_cast<PyArrayObject*>(ant_1_ids.ptr())->data);
	  m_p_ant_2_ids = reinterpret_cast<size_t*>(reinterpret_cast<PyArrayObject*>(ant_2_ids.ptr())->data);
	  m_p_measurement_time = reinterpret_cast<float*>(reinterpret_cast<PyArrayObject*>(measurement_time.ptr())->data);
	  m_p_channel_ref_freqs = reinterpret_cast<float*>(reinterpret_cast<PyArrayObject*>(channel_ref_freqs.ptr())->data);
	}
	//descriptors
	{
	  size_t rows = extract<size_t>(uvw.attr("shape")[0]);
	  size_t nchan = extract<size_t>(vis.attr("shape")[1]);
	  size_t ncorrs = extract<size_t>(vis.attr("shape")[2]);
	  m_num_rows = rows;
	  m_num_chan = nchan;
	  m_num_corrs = ncorrs;
	}
	//finally host and device allocate:
	checked_acquire_persistant();
	cudaSafeCall(cudaDeviceSynchronize());
      }
      uv_data(const uv_data & lvalue) = delete;
      uv_data(uv_data && rvalue){
	*this = std::move(rvalue);
      }
      uv_data& operator=(const uv_data & lvalue) = delete;
      uv_data& operator=(uv_data && rvalue) {
	m_resources_aquired = rvalue.m_resources_aquired;
	rvalue.m_resources_aquired = false;
	
	this->m_p_uvw = rvalue.m_p_uvw;
	this->m_p_vis = rvalue.m_p_vis;
	this->m_p_img_weights = rvalue.m_p_img_weights;
	this->m_p_flags = rvalue.m_p_flags;
	this->m_p_ant_1_ids = rvalue.m_p_ant_1_ids;
	this->m_p_ant_2_ids = rvalue.m_p_ant_2_ids;
	this->m_p_measurement_time = rvalue.m_p_measurement_time;
	this->m_p_channel_ref_freqs = rvalue.m_p_channel_ref_freqs;
	
	rvalue.m_p_uvw = nullptr;
	rvalue.m_p_vis = nullptr;
	rvalue.m_p_img_weights = nullptr;
	rvalue.m_p_flags = nullptr;
	rvalue.m_p_ant_1_ids = nullptr;
	rvalue.m_p_ant_2_ids = nullptr;
	rvalue.m_p_measurement_time = nullptr;
	rvalue.m_p_channel_ref_freqs = nullptr;
	
	m_d_uvw = std::move(rvalue.m_d_uvw);
	m_d_vis = std::move(rvalue.m_d_vis);
	m_d_img_weights = std::move(rvalue.m_d_img_weights);
	m_d_flags = std::move(rvalue.m_d_flags);
	m_d_ant_1_ids = std::move(rvalue.m_d_ant_1_ids);
	m_d_ant_2_ids = std::move(rvalue.m_d_ant_2_ids);
	m_d_measurement_time = std::move(rvalue.m_d_measurement_time);
	m_d_channel_ref_freqs = std::move(rvalue.m_d_channel_ref_freqs);
	cudaSafeCall(cudaDeviceSynchronize());
	return *this;
      }
      virtual ~uv_data(){
	checked_release_persistant();
      }
      uv_data & checked_acquire_persistant(){ //For use in Python context management try catch finally release mechanism (or with mechanism)
	if (!m_resources_aquired){
	  //Acquire resources here:
	  m_d_uvw = std::move(thrust::device_vector<uvwT>(m_p_uvw, m_p_uvw + m_num_rows));	  
	  size_t num_vis = m_num_rows * m_num_chan * m_num_corrs;
	  m_d_vis = std::move(thrust::device_vector<thrust::complex<float>>(m_p_vis, m_p_vis + num_vis));
	  size_t num_weights = m_num_chan * m_num_rows;
	  m_d_img_weights = std::move(thrust::device_vector<float>(m_p_img_weights, m_p_img_weights + num_weights));
	  m_d_flags = std::move(thrust::device_vector<bool>(m_p_flags, m_p_flags + num_weights));
	  m_d_ant_1_ids = std::move(thrust::device_vector<size_t>(m_p_ant_1_ids,m_p_ant_1_ids + m_num_rows));
	  m_d_ant_2_ids = std::move(thrust::device_vector<size_t>(m_p_ant_2_ids,m_p_ant_2_ids + m_num_rows));  
	  m_d_measurement_time = std::move(thrust::device_vector<float>(m_p_measurement_time, m_p_measurement_time + m_num_rows));
	  m_d_channel_ref_freqs = std::move(thrust::device_vector<float>(m_p_channel_ref_freqs, m_p_channel_ref_freqs + m_num_chan));
	}
	m_resources_aquired = true;
	return *this;
      }
      void checked_release_persistant(boost::python::object except_type = boost::python::object(),
				      boost::python::object exception_value = boost::python::object(),
				      boost::python::object traceback = boost::python::object()){ //For use in Python context management try catch finally release mechanism
	if (m_resources_aquired){	  
	  //not responsible for the raw pointers passed in originally:
	  m_p_uvw = nullptr;
	  m_p_vis = nullptr;
	  m_p_img_weights = nullptr;
	  m_p_flags = nullptr;
	  m_p_ant_1_ids = nullptr;
	  m_p_ant_2_ids = nullptr;
	  m_p_measurement_time = nullptr;
	  m_p_channel_ref_freqs = nullptr;
	  
	  #define release_array(arr) {arr.clear(); arr.shrink_to_fit();}
	  release_array(m_d_uvw);
	  release_array(m_d_vis);
	  release_array(m_d_img_weights);
	  release_array(m_d_flags);
	  release_array(m_d_ant_1_ids);
	  release_array(m_d_ant_2_ids);
	  release_array(m_d_measurement_time);
	  release_array(m_d_channel_ref_freqs);
	  cudaSafeCall(cudaDeviceSynchronize());
	}
	m_resources_aquired = false;
      }
      /**
       * Copies device uv data to host (after previous kernels have finished)
       */
      void sync_device2host(){
	cudaSafeCall(cudaDeviceSynchronize());
	size_t num_vis = m_num_rows * m_num_chan * m_num_corrs;
	size_t num_weights = m_num_chan * m_num_rows;
	thrust::copy_n(m_d_uvw.begin(),m_num_rows,m_p_uvw);
	thrust::copy_n(m_d_vis.begin(),num_vis,m_p_vis);
	thrust::copy_n(m_d_img_weights.begin(),num_weights,m_p_img_weights);
	thrust::copy_n(m_d_flags.begin(),num_weights,m_p_flags);
	thrust::copy_n(m_d_ant_1_ids.begin(),m_num_rows,m_p_ant_1_ids);
	thrust::copy_n(m_d_ant_2_ids.begin(),m_num_rows,m_p_ant_2_ids);
	thrust::copy_n(m_d_measurement_time.begin(),m_num_rows,m_p_measurement_time);
	thrust::copy_n(m_d_channel_ref_freqs.begin(),m_num_chan,m_p_channel_ref_freqs);
	cudaSafeCall(cudaDeviceSynchronize());
      }
      /**
       * Copies host uv data to host (after previous kernels have finished)
       */
      void sync_host2device(){
	cudaSafeCall(cudaDeviceSynchronize());
	size_t num_vis = m_num_rows * m_num_chan * m_num_corrs;
	size_t num_weights = m_num_chan * m_num_rows;
	thrust::copy_n(m_p_uvw,m_num_rows,m_d_uvw.begin());
	thrust::copy_n(m_p_vis,num_vis,m_d_vis.begin());
	thrust::copy_n(m_p_img_weights,num_weights,m_d_img_weights.begin());
	thrust::copy_n(m_p_flags,num_weights,m_d_flags.begin());
	thrust::copy_n(m_p_ant_1_ids,m_num_rows,m_d_ant_1_ids.begin());
	thrust::copy_n(m_p_ant_2_ids,m_num_rows,m_d_ant_2_ids.begin());
	thrust::copy_n(m_p_measurement_time,m_num_rows,m_d_measurement_time.begin());
	thrust::copy_n(m_p_channel_ref_freqs,m_num_chan,m_d_channel_ref_freqs.begin());
	cudaSafeCall(cudaDeviceSynchronize());
      }
    };
  }
}