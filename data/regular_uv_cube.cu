#include "regular_uv_cube.hpp"

BOOST_PYTHON_MODULE(regular_uv_cube)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  boost::python::class_<imaging_core::data::regular_uv_cube<float>,
			boost::noncopyable>("regular_uv_cube", 
					    boost::python::init<boost::python::numeric::array &>())
					    .def("__enter__",
						&imaging_core::data::regular_uv_cube<float>::checked_acquire_persistant,
						boost::python::return_value_policy<boost::python::reference_existing_object>())
					    .def("__exit__", &imaging_core::data::regular_uv_cube<float>::checked_release_persistant)
					    .def("checked_release_persistant",
						&imaging_core::data::regular_uv_cube<float>::checked_release_persistant,
						(boost::python::arg("except_type")=boost::python::api::object(),
						  boost::python::arg("exception_value")=boost::python::api::object(),
						  boost::python::arg("traceback")=boost::python::api::object()))
					    .def("sync_host2device",&imaging_core::data::regular_uv_cube<float>::sync_host2device)
					    .def("sync_device2host",&imaging_core::data::regular_uv_cube<float>::sync_device2host)
					    ;
}
