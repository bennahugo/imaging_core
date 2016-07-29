#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
include(CMakePythonAddCudaModule)
PYTHON_ADD_CUDA_MODULE(bda_resampler bda_resampler.cu)
target_link_libraries(bda_resampler ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_subdirectory(bda_traits_policies)