#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
#FILE(COPY test_import.py DESTINATION .)
include(CMakeCopyIfDifferent)
SET(TEST_SCRIPTS test_import_uv_data.py test_import_uv_cube.py)
COPY_IF_DIFFERENT(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} "${TEST_SCRIPTS}" IncludeTargets "Includes")
add_custom_target(CopyTests ALL DEPENDS ${IncludeTargets})
ADD_TEST(01-NonRegUVDATA-Import tests/test_import_uv_data.py)
ADD_TEST(02-RegUVDATA-Import tests/test_import_uv_cube.py)