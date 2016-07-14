#include "uv_data.hpp"

BOOST_PYTHON_MODULE(uv_data)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  boost::python::class_<imaging_core::resamplers::uv_data>("uv_data", 
							   boost::python::init<
							      boost::python::numeric::array &
							   >());
}