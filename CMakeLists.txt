project(imaging_core)
cmake_minimum_required(VERSION 2.8)

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


ENABLE_TESTING()
#INSTALL(TARGETS imaging_core RUNTIME DESTINATION bin)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} 
		    ${PYTHON_INCLUDE_DIRS} 
		    ${PYTHON_NUMPY_INCLUDE_DIR} 
		    ${PYTHON_NUMPY_INCLUDE_DIR}/numpy
		    ${CUDA_INCLUDE_DIRECTORIES})

#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(tests)
add_subdirectory(resamplers)