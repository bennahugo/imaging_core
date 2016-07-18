#include "bda_resampler.hpp"
BOOST_PYTHON_MODULE(bda_resampler)
{
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    
  boost::python::class_<imaging_core::resamplers::a_policy>("a_policy",boost::python::no_init)
							    .def("print",&imaging_core::resamplers::a_policy::print)
							    .staticmethod("print")
							    ;
  boost::python::class_<imaging_core::resamplers::b_policy>("b_policy",boost::python::no_init)
							    .def("print",&imaging_core::resamplers::b_policy::print)
							    .staticmethod("print")
							    ;
  #define bda_resampler_wrapper(policy) \
  boost::python::class_<imaging_core::resamplers::bda_resampler<policy>, \
			boost::noncopyable>("bda_resampler_"#policy, \
					    boost::python::init<size_t, size_t, size_t, size_t, \
								size_t, size_t, size_t>()) \
					    .def("__enter__", \
						&imaging_core::resamplers::bda_resampler<policy>::checked_acquire_persistant, \
						boost::python::return_value_policy<boost::python::reference_existing_object>()) \
					    .def("__exit__", &imaging_core::resamplers::bda_resampler<policy>::checked_release_persistant) \
					    .def("checked_release_persistant", \
						&imaging_core::resamplers::bda_resampler<policy>::checked_release_persistant, \
						(boost::python::arg("except_type")=boost::python::api::object(), \
						  boost::python::arg("exception_value")=boost::python::api::object(), \
						  boost::python::arg("traceback")=boost::python::api::object())) \
					    .def("nonregular2regular", &imaging_core::resamplers::bda_resampler<policy>::nonregular2regular) \
					    .def("regular2nonregular", &imaging_core::resamplers::bda_resampler<policy>::regular2nonregular) \
					    ; 
 {
    using namespace imaging_core::resamplers;
    bda_resampler_wrapper(a_policy)
    bda_resampler_wrapper(b_policy)
 }
}