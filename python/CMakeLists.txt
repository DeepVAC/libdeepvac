# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

file(GLOB CPP4PYTHON_LIST src/*.cpp)
message(STATUS "found CPP4PYTHON_LIST: " ${CPP4PYTHON_LIST})

find_package(pybind11 REQUIRED)
pybind11_add_module(pydeepvac ${CPP4PYTHON_LIST})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

target_link_libraries( pydeepvac "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY} ${OpenCV_LIBS} deepvac)

target_include_directories(pydeepvac PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils import sysconfig as sc;print(sc.get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_SITE: ${PYTHON_SITE}")
 
install(TARGETS pydeepvac
    COMPONENT python
    LIBRARY DESTINATION ${PYTHON_SITE})