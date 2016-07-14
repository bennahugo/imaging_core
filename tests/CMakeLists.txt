#Ensure __init__.py is added to out-of-source build directory
execute_process(COMMAND touch __init__.py
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY test_import.py DESTINATION .)
ADD_TEST(01-Gridder-Import tests/test_import.py)