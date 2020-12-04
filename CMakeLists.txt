# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(DEEPVAC_VERSION 0.1.0)
project(deepvac
	VERSION ${DEEPVAC_VERSION}
    DESCRIPTION "A library for using PyTorch model in C++."
    HOMEPAGE_URL "https://github.com/deepvac/libdeepvac"
    LANGUAGES CXX)

option(GARRULOUS_GEMFIELD "Build deepvac libraries garrulously" OFF)
option(BUILD_STATIC "build static library" OFF)
option(USE_CUDA "Use CUDA" OFF)


if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LINUX TRUE)
else()
    set(LINUX FALSE)
endif()

if(NOT LINUX)
    message(FATAL_ERROR "deepvac does not support platform: " ${CMAKE_SYSTEM_NAME})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

#setup library output dir
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#setup binary output dir
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

#replace with your libtorch dir
#set(CMAKE_PREFIX_PATH /home/gemfield/libtorch/)
if(GARRULOUS_GEMFIELD)
    message(STATUS "will build deepvac library garrulously...")
    add_definitions(-DGARRULOUS_GEMFIELD)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

macro(append_lib_if_found)
    foreach (_arg ${ARGN})
        find_library(${_arg}_LIBRARY ${_arg} PATHS "${TORCH_INSTALL_PREFIX}/lib")
        if(${_arg}_LIBRARY)
	    list(APPEND TORCH_STATIC_LIBRARIES ${${_arg}_LIBRARY})
        else()
	    message(WARNING "static library ${${_arg}_LIBRARY} not found.")
        endif()
    endforeach()
endmacro()


#libtorch static stuff
if(BUILD_STATIC)
    message(STATUS "will find libtorch static library since BUILD_STATIC is ON")

    #libtorch.a
    find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

    #libtorch_cpu.a
    find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
    find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

    SET (TORCH_STATIC_LIBRARIES "-Wl,--whole-archive ${TORCH_LIBRARY} ${TORCH_CPU_LIBRARY} ${TORCH_CUDA_LIBRARY} -Wl,--no-whole-archive")

    append_lib_if_found(c10 c10_cuda nnpack pytorch_qnnpack XNNPACK pthreadpool eigen_blas cpuinfo clog)
    append_lib_if_found(caffe2_protos protobuf-lite protobuf protoc)
    append_lib_if_found(dnnl fbgemm mkldnn sleef asmjit)
    append_lib_if_found(onnx onnx_proto)
    append_lib_if_found(nccl_static)
    message(STATUS "found libtorch static library: ${TORCH_STATIC_LIBRARIES}")
    if(USE_CUDA)
        message(STATUS "will build with CUDA")
        set(CUDA_LIBRARIES "-lcusparse -lcufft -lcufftw -lcupti -lcusolver -lcurand -lcublas -lcublasLt -lnvToolsExt -lcudnn -lcuda -lnvrtc -lcudart")
    endif()
endif()


macro (add_syszux_sources)
    file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        list (APPEND SRC_LIST "${_src}")
    endforeach()
    if (_relPath)
        set (SRC_LIST ${SRC_LIST} PARENT_SCOPE)
    endif()
endmacro()

macro (add_header_dir)
    foreach (_header_dir ${ARGN})
        list (APPEND HEADER_DIR_LIST "${_header_dir}")
    endforeach()
    set (HEADER_DIR_LIST ${HEADER_DIR_LIST} PARENT_SCOPE)
endmacro()

macro (add_syszux_headers)
    file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        list (APPEND HEADER_FILE_LIST "${_src}")
    endforeach()
    if (_relPath)
        set (HEADER_FILE_LIST ${HEADER_FILE_LIST} PARENT_SCOPE)
    endif()
endmacro()

macro (add_syszux_flags)
    foreach (_src ${ARGN})
        list (APPEND FLAGS_FILE_LIST "${_src}")
    endforeach()
    set (FLAGS_FILE_LIST ${FLAGS_FILE_LIST} PARENT_SCOPE)
endmacro()

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_HEADER: ${TORCH_INCLUDE_DIRS}")

# add subdiretories
add_subdirectory(utils)
add_subdirectory(loader)
add_subdirectory(module)
add_subdirectory(service)

#add library
if(BUILD_STATIC)
    message(STATUS "will build libdeepvac static library")
    add_library(deepvac STATIC ${SRC_LIST})
    target_compile_options(deepvac PUBLIC -fopenmp)
else()
    message(STATUS "will build libdeepvac shared library")
    add_library(deepvac SHARED ${SRC_LIST})
endif()

target_include_directories(deepvac PUBLIC 
    "$<INSTALL_INTERFACE:include/deepvac>"   
    "$<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS};${HEADER_DIR_LIST};${OpenCV_INCLUDE_DIRS}>"
)

set_target_properties(deepvac PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(deepvac PROPERTIES LINK_FLAGS_RELEASE -s)
target_compile_options(deepvac PRIVATE -Werror)

foreach(_src ${FLAGS_FILE_LIST})
    string(REGEX MATCH "^[^@]+" def_name ${_src})
    string(REPLACE "_" "." def_value ${def_name})
    string(REPLACE "${def_name}@" "" src_name ${_src})
    set_property(
        SOURCE ${src_name}
        APPEND
        PROPERTY COMPILE_DEFINITIONS
        ${def_name}="${CMAKE_INSTALL_PREFIX}/lib/deepvac/${def_value}"
    )
endforeach()

find_program(STRIP_COMMAND NAMES "strip")
if(STRIP_COMMAND AND (NOT BUILD_STATIC))
    message(STATUS "found strip command...")
    add_custom_command(TARGET deepvac POST_BUILD COMMAND strip ${LIBRARY_OUTPUT_PATH}/lib*.*)
endif()

add_subdirectory(python)
add_subdirectory(examples)

#config helper
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/DeepvacConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/DeepvacConfigVersion.cmake
    @ONLY)
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/DeepvacConfig.cmake.in
    ${PROJECT_BINARY_DIR}/DeepvacConfig.cmake
    @ONLY)
install(FILES
    ${PROJECT_BINARY_DIR}/DeepvacConfigVersion.cmake
    ${PROJECT_BINARY_DIR}/DeepvacConfig.cmake
    DESTINATION share/deepvac)
#
message(STATUS "install headers: " ${HEADER_FILE_LIST})

install(TARGETS deepvac EXPORT deepvac DESTINATION "lib")
install(FILES ${HEADER_FILE_LIST} DESTINATION "include/deepvac")

install(EXPORT deepvac DESTINATION "share/deepvac")
install(FILES ${DEEPVAC_FILE_LIST} DESTINATION "lib/deepvac") 
