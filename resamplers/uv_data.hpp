#pragma once
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <numpy/ndarrayobject.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
namespace imaging_core{
  namespace resamplers{
    using namespace std;
    using namespace boost::python;
    
    struct uvwT {
      float m_u,m_v,m_w;
      __device__ __host__ uvwT(float u=0, float v=0, float w=0): m_u(u),m_v(v),m_w(w){}
    };
    
    struct uv_data
    {    
      bool m_resources_aquired;
      //Raw pointers: do not assume responsibility for these:
      uvwT * m_p_uvw;
      thrust::complex<float> * m_p_vis;
      bool * m_p_flags;
      float * m_p_img_weights;
      size_t * m_p_ant_1_ids;
      size_t * m_p_ant_2_ids;
      float * m_p_measurement_time;
      float * m_p_channel_ref_freqs;
      //Host vectors:
      thrust::host_vector<uvwT> m_h_uvw;
      thrust::host_vector<thrust::complex<float>> m_h_vis;
      //Device vectors:
      thrust::device_vector<uvwT> m_d_uvw;
      thrust::device_vector<thrust::complex<float>> m_d_vis;
      size_t m_num_rows;
      size_t m_num_chan;
      size_t m_num_corrs;
    
      uv_data(numeric::array & uvw,
	      numeric::array & vis,
	      numeric::array & flags,
	      numeric::array & weights,
	      numeric::array & ant_1_ids,
	      numeric::array & ant_2_ids,
	      numeric::array & measurement_time,
	      numeric::array & channel_ref_freqs):m_resources_aquired(false){
	/*
	size_t l = len(uvw.attr("shape"));
	
	double u = extract<double>(uvw[boost::python::make_tuple(0,0)]);
	double v = extract<double>(uvw[boost::python::make_tuple(0,1)]);
	double w = extract<double>(uvw[boost::python::make_tuple(0,2)]);
	bool cont = uvw.attr("dtype");
	cout << cont << endl;
	cout << l << endl;
	cout << rows << endl;
	cout << u << " " << v << " " << w << endl;
	*/
	//Check that the arrays are contiguous c arrays
	{
// 	  if (!std::reinterpret_cast<PyArrayFlagsObject*>(uvw.attr("flags") & NPY_C_C))
// 	    throw std::runtime_error("UVW array must be contiguous array");
	}
	//Check dimensions of the arrays
	{
	  size_t nd_uvw = len(uvw.attr("shape"));
	  size_t nd_vis = len(vis.attr("shape"));
	  size_t nd_flags = len(flags.attr("shape"));
	  size_t nd_ant_1_ids = len(ant_1_ids.attr("shape"));
	  size_t nd_ant_2_ids = len(ant_2_ids.attr("shape"));
	  size_t nd_measurement_time = len(measurement_time.attr("shape"));
	  size_t nd_channel_ref_freqs = len(channel_ref_freqs.attr("shape"));
	}
	
	size_t rows = extract<size_t>(uvw.attr("shape")[0]);
	size_t nchan = extract<size_t>(vis.attr("shape")[1]);
	size_t ncorrs = extract<size_t>(vis.attr("shape")[2]);
	m_p_uvw = reinterpret_cast<uvwT *>(reinterpret_cast<PyArrayObject*>(uvw.ptr())->data);
	m_p_vis = reinterpret_cast<thrust::complex<float> *>(reinterpret_cast<PyArrayObject*>(vis.ptr())->data);
	m_num_rows = rows;
	m_num_chan = nchan;
	m_num_corrs = ncorrs;
	
	//finally host and device allocate:
	checked_acquire_persistant();
      }
      ~uv_data(){
	checked_release_persistant();
      }
      uv_data & checked_acquire_persistant(){ //For use in Python context management try catch finally release mechanism (or with mechanism)
	if (!m_resources_aquired){
	  //Acquire resources here:
	  using namespace thrust::system::cuda::experimental;
	  m_h_uvw = std::move(thrust::host_vector<uvwT, 
			      pinned_allocator<uvwT> >(m_p_uvw, m_p_uvw + m_num_rows));
	  m_d_uvw = m_h_uvw;
	  
	  std::size_t num_vis = m_num_rows * m_num_chan * m_num_corrs;
	  m_h_vis = std::move(thrust::host_vector<thrust::complex<float>,
			      pinned_allocator<thrust::complex<float>>>(m_p_vis, m_p_vis + num_vis));
	  m_d_vis = m_h_vis;
	}
	m_resources_aquired = true;
	return *this;
      }
      
      void checked_release_persistant(boost::python::object except_type = boost::python::object(),
				      boost::python::object exception_value = boost::python::object(),
				      boost::python::object traceback = boost::python::object()){ //For use in Python context management try catch finally release mechanism
	if (m_resources_aquired){
	  m_h_uvw.clear();
	  m_h_uvw.shrink_to_fit();
	  m_d_uvw.clear();
	  m_d_uvw.shrink_to_fit();
	}
	m_resources_aquired = false;
      }
    };
  }
}