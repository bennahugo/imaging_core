#pragma once

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include "uv_data.hpp"
#include "bda_traits_policies/correlation_gridding_policy.hpp"

namespace imaging_core{
 
  namespace resamplers{
    using namespace boost::python;
    using namespace std;        
    
    template <typename active_correlation_gridding_policy>
    class bda_resampler {  
    private:
      bool m_resources_aquired;
    public:
      bda_resampler():m_resources_aquired(false){
	checked_acquire_persistant(); //Ensure RAII is followed in pure C++ calls
      }
      bda_resampler(bda_resampler && rvalue):
			m_resources_aquired(rvalue.m_resources_aquired){
	rvalue.m_resources_aquired = false;
      }
      ~bda_resampler(){
	checked_release_persistant(); //Ensure RAII is followed in pure C++ calls
      }
      bda_resampler(bda_resampler & lvalue) = delete; //ill-defined for persistant resources
      
      
      bda_resampler & checked_acquire_persistant(){ //For use in Python context management try catch finally release mechanism (or with mechanism)
	if (!m_resources_aquired){
	  //Acquire resources here
	}
	m_resources_aquired = true;
	return *this;
      }
      
      void checked_release_persistant(boost::python::object except_type = boost::python::object(),
				      boost::python::object exception_value = boost::python::object(),
				      boost::python::object traceback = boost::python::object()){ //For use in Python context management try catch finally release mechanism
	if (m_resources_aquired){
	  //Release resources here
	}
	m_resources_aquired = false;
      }
      void nonregular2regular(uv_data & input_uv,
			      bool append = true){
	cout << "Grid stub" << endl;
      }
      void regular2nonregular(uv_data & output_uv,
			      bool append = true){
	cout << "DeGrid stub" << endl;
      }
    };
  }
}																		\

