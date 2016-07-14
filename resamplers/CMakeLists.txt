#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
PYTHON_ADD_MODULE(gridder gridder.cpp)
PYTHON_ADD_MODULE(uv_data uv_data.cpp)
