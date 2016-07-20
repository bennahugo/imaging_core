#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
include(CMakePythonAddCudaModule)
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions("-std=c++11")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_21") 
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
list(APPEND CUDA_NVCC_FLAGS "-Xptxas -dlcm=ca -std=c++11")

PYTHON_ADD_CUDA_MODULE(bda_resampler bda_resampler.cu)
target_link_libraries(bda_resampler ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
PYTHON_ADD_CUDA_MODULE(uv_data uv_data.cu)
target_link_libraries(uv_data ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_subdirectory(bda_traits_policies)