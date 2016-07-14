#include "gridder.hpp"
BOOST_PYTHON_MODULE(gridder)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  boost::python::def("grid", imaging_core::resamplers::grid);
}