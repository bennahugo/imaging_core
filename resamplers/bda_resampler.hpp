#pragma once

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>

#include "uv_data.hpp"

namespace imaging_core{
 
  namespace resamplers{
    using namespace boost::python;
    using namespace std;    
    class a_policy { public: static void print(){cout << "a is active" << endl;} };
    class b_policy { public: static void print(){cout << "b is active" << endl;} };
    
    template <typename policy>
    class bda_resampler {
    private:
      size_t m_resources_aquired;
      size_t m_uv_grid_u_size;
      size_t m_uv_grid_v_size;
      size_t m_uv_grid_channels;
      size_t m_uv_grid_polarizations;
      size_t m_max_non_regular_uv_samples;
      size_t m_max_channels_per_uv_sample;
      size_t m_num_uv_sample_correlations;
    public:
      bda_resampler(size_t uv_grid_u_size, 
		    size_t uv_grid_v_size,
		    size_t uv_grid_channels,
		    size_t uv_grid_polarizations,
		    size_t max_non_regular_uv_samples,
		    size_t max_channels_per_uv_sample,
		    size_t num_uv_sample_correlations):
			m_uv_grid_u_size(uv_grid_u_size),
			m_uv_grid_v_size(uv_grid_v_size),
			m_uv_grid_channels(uv_grid_channels),
			m_uv_grid_polarizations(uv_grid_polarizations),
			m_max_non_regular_uv_samples(max_non_regular_uv_samples),
			m_max_channels_per_uv_sample(max_channels_per_uv_sample),
			m_num_uv_sample_correlations(num_uv_sample_correlations){
	checked_acquire_persistant(); //Ensure RAII is followed in pure C++ calls
	policy::print();
      }
      bda_resampler(bda_resampler && rvalue):
			m_resources_aquired(rvalue.m_resources_aquired){
	rvalue.m_resources_aquired = false;
      }
      ~bda_resampler(){
	checked_release_persistant(); //Ensure RAII is followed in pure C++ calls
      }
      bda_resampler(bda_resampler & lvalue) = delete; //ill-defined for persistant resources
      
      
      bda_resampler & checked_acquire_persistant(){ //For use in Python context management try catch finally release mechanism
	if (!m_resources_aquired){
	   
	}
	m_resources_aquired = true;
	
	return *this;
      }
      
      void checked_release_persistant(boost::python::object except_type = boost::python::object(),
				      boost::python::object exception_value = boost::python::object(),
				      boost::python::object traceback = boost::python::object()){ //For use in Python context management try catch finally release mechanism
	if (m_resources_aquired){
	  
	}
	m_resources_aquired = false;
      }
      
      void nonregular2regular(uv_data & input_uv){
	cout << input_uv.m_uvw[0] << input_uv.m_uvw[1] << input_uv.m_uvw[2] << endl;
      }
      void regular2nonregular(uv_data & output_uv){
      }
    };
  }
}																		\

