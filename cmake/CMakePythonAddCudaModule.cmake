# PYTHON_ADD_CUDA_MODULE(<name> src1 src2 ... srcN) is used to build cuda modules for python.
# PYTHON_WRITE_MODULES_HEADER(<filename>) writes a header file you can include
# in your sources to initialize the static python modules
# This method extends kitware's FindPython.cmake to include libraries compiled with
# the NVIDIA nvcc compiler
function(PYTHON_ADD_CUDA_MODULE _NAME )
  get_property(_TARGET_SUPPORTS_SHARED_LIBS
    GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
  option(PYTHON_ENABLE_MODULE_${_NAME} "Add module ${_NAME}" TRUE)
  option(PYTHON_MODULE_${_NAME}_BUILD_SHARED
    "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

  # Mark these options as advanced
  mark_as_advanced(PYTHON_ENABLE_MODULE_${_NAME}
    PYTHON_MODULE_${_NAME}_BUILD_SHARED)

  if(PYTHON_ENABLE_MODULE_${_NAME})
    if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      set(PY_MODULE_TYPE MODULE)
    else()
      set(PY_MODULE_TYPE STATIC)
      set_property(GLOBAL  APPEND  PROPERTY  PY_STATIC_MODULES_LIST ${_NAME})
    endif()

    set_property(GLOBAL  APPEND  PROPERTY  PY_MODULES_LIST ${_NAME})
    cuda_add_library(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
#    target_link_libraries(${_NAME} ${PYTHON_LIBRARIES})

    if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      set_target_properties(${_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
      if(WIN32 AND NOT CYGWIN)
        set_target_properties(${_NAME} PROPERTIES SUFFIX ".pyd")
      endif()
    endif()

  endif()
endfunction()