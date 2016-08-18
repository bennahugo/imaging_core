#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
include(CMakePythonAddCudaModule)

PYTHON_ADD_CUDA_MODULE(irregular_uv_data irregular_uv_data.cu)
target_link_libraries(irregular_uv_data ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
PYTHON_ADD_CUDA_MODULE(regular_uv_cube regular_uv_cube.cu)
target_link_libraries(regular_uv_cube ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})