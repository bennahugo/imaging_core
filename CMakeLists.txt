project(imaging_core)
cmake_minimum_required(VERSION 2.8)
#--------------------------------------------------------------------------------------------------------------------------
#Required Packages
#--------------------------------------------------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(NumPy REQUIRED)
if (NOT ${BOOST_FOUND})
  message(FATAL_ERROR "Failed to find Boost.")
endif (NOT ${BOOST_FOUND})
if (NOT ${PYTHON_FOUND})
  message(FATAL_ERROR "Failed to find Python.")
endif (NOT ${PYTHON_FOUND})
if (NOT ${NUMPY_FOUND})
  message(FATAL_ERROR "Failed to find Numpy.")
endif (NOT ${NUMPY_FOUND})
if (NOT ${CUDA_FOUND})
  message(FATAL_ERROR "Failed to find CUDA.")
endif (NOT ${CUDA_FOUND})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} 
		    ${PYTHON_INCLUDE_DIRS} 
		    ${PYTHON_NUMPY_INCLUDE_DIR} 
		    ${PYTHON_NUMPY_INCLUDE_DIR}/numpy
		    ${CUDA_INCLUDE_DIRECTORIES})

#--------------------------------------------------------------------------------------------------------------------------
#Project-wide configuration
#--------------------------------------------------------------------------------------------------------------------------
ENABLE_TESTING()
#INSTALL(TARGETS imaging_core RUNTIME DESTINATION bin)

add_definitions("-std=c++11")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_21") 
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
list(APPEND CUDA_NVCC_FLAGS "-Xptxas -dlcm=ca -std=c++11")

#--------------------------------------------------------------------------------------------------------------------------
#Include Project submodules
#--------------------------------------------------------------------------------------------------------------------------
#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(resamplers data)

add_subdirectory(tests)
add_subdirectory(resamplers)
add_subdirectory(data)

