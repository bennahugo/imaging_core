#include "uv_data.hpp"

BOOST_PYTHON_MODULE(uv_data)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  boost::python::class_<imaging_core::resamplers::uv_data,
			boost::noncopyable>("uv_data", 
					    boost::python::init<boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &,
								boost::python::numeric::array &>())
					    .def("__enter__",
						&imaging_core::resamplers::uv_data::checked_acquire_persistant,
						boost::python::return_value_policy<boost::python::reference_existing_object>())
					    .def("__exit__", &imaging_core::resamplers::uv_data::checked_release_persistant)
					    .def("checked_release_persistant",
						&imaging_core::resamplers::uv_data::checked_release_persistant,
						(boost::python::arg("except_type")=boost::python::api::object(),
						  boost::python::arg("exception_value")=boost::python::api::object(),
						  boost::python::arg("traceback")=boost::python::api::object()))
					    .def("sync_host2device",&imaging_core::resamplers::uv_data::sync_host2device)
					    .def("sync_device2host",&imaging_core::resamplers::uv_data::sync_device2host)
					    ;
}