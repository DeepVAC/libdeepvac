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
option(BUILD_STATIC "build static libdeepvac library" ON)
option(USE_MKL "use MKL as pytorch LAPACK backend" OFF)
option(USE_LOADER "build loader which needs c++17" OFF)
option(USE_CUDA "Use CUDA" OFF)
option(USE_NUMA "Use NUMA" OFF)
option(USE_TENSORRT "Use TensorRT" OFF)
option(BUILD_ALL_EXAMPLES "build all examples" OFF)
#turn off USE_STATIC_LIBTORCH is not tested.
option(USE_STATIC_LIBTORCH "use LibTorch static library" ON)
#You should customize the blas dependencies.

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LINUX TRUE)
else()
    set(LINUX FALSE)
endif()

if(NOT LINUX)
    message(FATAL_ERROR "deepvac does not support platform: " ${CMAKE_SYSTEM_NAME})
endif()

if(USE_MKL AND USE_STATIC_LIBTORCH)
    if(NOT DEFINED MKL_HOME)
        set(MKL_HOME "/opt/intel/mkl")
    endif()
    if(NOT EXISTS ${MKL_HOME})
        message(FATAL "unable to locate MKL library, either set MKL_HOME or install MKL")
    endif()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MKL_ARCHITECTURES intel64)
    else()
        set(MKL_ARCHITECTURES ia32)
    endif()
    #must consistent with pytorch static library.
    set(INT_LIB "libmkl_intel_lp64.a")
    set(SEQ_LIB "libmkl_sequential.a")
    #set(THR_LIB "libmkl_intel_thread.a")
    set(THR_LIB "libmkl_gnu_thread.a")
    set(COR_LIB "libmkl_core.a")
    
    find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include)
    
    set(MKL_LIB_PATH ${MKL_HOME}/lib/${MKL_ARCHITECTURES})
    find_library(MKL_INTERFACE_LIBRARY
                 NAMES ${INT_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
                 NAMES ${SEQ_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_THREAD_LAYER_LIBRARY
                 NAMES ${THR_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_CORE_LIBRARY
                 NAMES ${COR_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_CORE_LIBRARY} ${MKL_THREAD_LAYER_LIBRARY} ${MKL_INTERFACE_LIBRARY} )
    set(MKL_LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
endif()

set(CMAKE_CXX_STANDARD 17)
#setup library output dir
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#setup binary output dir
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
#replace with your libtorch dir
#set(CMAKE_PREFIX_PATH /home/gemfield/libtorch/)
if(GARRULOUS_GEMFIELD)
    message(STATUS "will build deepvac library garrulously...")
    add_definitions(-DGARRULOUS_GEMFIELD)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 QUIET)

if(USE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES include)
    message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
    find_library(TENSORRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)
    find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)

    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
    MESSAGE(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")
    find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
    if(NOT TENSORRT_FOUND)
        message(ERROR "Cannot find TensorRT library.")
    endif()
endif()

#libtorch static stuff
if(USE_STATIC_LIBTORCH)
    message(STATUS "will find libtorch static library since USE_STATIC_LIBTORCH is ON")
    list(APPEND LIBTORCH_CUDA_LIBRARIES ${TORCH_LIBRARIES})
    if(USE_MKL)
        list(APPEND LIBTORCH_CUDA_LIBRARIES ${MKL_LIBRARIES})
        list(APPEND LIBTORCH_CUDA_LIBRARIES "-lpthread")
        list(APPEND LIBTORCH_CUDA_LIBRARIES "-lm")
        list(APPEND LIBTORCH_CUDA_LIBRARIES "-ldl")
    endif()
    list(APPEND LIBTORCH_CUDA_LIBRARIES "${SYSTEM_LAPACK_LIBRARIES}")

    foreach(GEM ${LIBTORCH_CUDA_LIBRARIES})
        string(FIND ${GEM} "_cuda.a" GEM_FIELD)
        if(NOT ${GEM_FIELD} EQUAL "-1")
            message(STATUS "found cuda library need to be removed from LIBTORCH_CPU_LIBRARIES: ${GEM}")
            continue()
        endif()
        list(APPEND LIBTORCH_CPU_LIBRARIES ${GEM})
    endforeach()

    if(USE_CUDA)
        find_package(CUDA REQUIRED)
        message(STATUS "will build with CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
        find_library(CUDNN_LIBRARY cudnn
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

        find_library(cublas_LIBRARY cublas
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

        find_library(cublasLT_LIBRARY cublasLt
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

        list(APPEND LIBCUDA_LIBRARIES
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse.so
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusolver.so
            ${cublas_LIBRARY}
            ${cublasLT_LIBRARY}
            ${CUDA_nvToolsExt_LIBRARY}
            ${CUDNN_LIBRARY}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
        )
        if(USE_NUMA)
            list(APPEND LIBCUDA_LIBRARIES -lnuma)
        endif()
        message(STATUS "LIBCUDA_LIBRARIES: ${LIBCUDA_LIBRARIES}")
        set(LIBTORCH_DEFAULT_LIBRARIES ${LIBTORCH_CUDA_LIBRARIES})
    else()
        set(LIBTORCH_DEFAULT_LIBRARIES ${LIBTORCH_CPU_LIBRARIES})
    endif()
    message(STATUS "LIBTORCH_CUDA_LIBRARIES: ${LIBTORCH_CUDA_LIBRARIES}")
    message(STATUS "LIBTORCH_CPU_LIBRARIES: ${LIBTORCH_CPU_LIBRARIES}")
else()
    message(FATAL "Turn off USE_STATIC_LIBTORCH is not tested yet.")
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

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_HEADER: ${TORCH_INCLUDE_DIRS}")

set(DEEPVAC_LIBTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
set(DEEPVAC_TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(DEEPVAC_CV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})

set(DEEPVAC_LIBRARIES deepvac)
set(DEEPVAC_LIBTORCH_CPU_LIBRARIES ${LIBTORCH_CPU_LIBRARIES})
set(DEEPVAC_LIBTORCH_CUDA_LIBRARIES ${LIBTORCH_CUDA_LIBRARIES})
set(DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES ${LIBTORCH_DEFAULT_LIBRARIES})
set(DEEPVAC_LIBCUDA_LIBRARIES ${LIBCUDA_LIBRARIES})
set(DEEPVAC_TENSORRT_LIBRARIES ${TENSORRT_LIBRARY})
set(DEEPVAC_CV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "DEEPVAC_LIBTORCH_INCLUDE_DIRS: ${DEEPVAC_LIBTORCH_INCLUDE_DIRS}")
message(STATUS "DEEPVAC_TENSORRT_INCLUDE_DIRS: ${DEEPVAC_TENSORRT_INCLUDE_DIRS}")
message(STATUS "DEEPVAC_CV_INCLUDE_DIRS: ${DEEPVAC_CV_INCLUDE_DIRS}")

message(STATUS "DEEPVAC_LIBRARIES: ${DEEPVAC_LIBRARIES}")
message(STATUS "DEEPVAC_LIBTORCH_CPU_LIBRARIES: ${DEEPVAC_LIBTORCH_CPU_LIBRARIES}")
message(STATUS "DEEPVAC_LIBTORCH_CUDA_LIBRARIES: ${DEEPVAC_LIBTORCH_CUDA_LIBRARIES}")
message(STATUS "DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES: ${DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES}")
message(STATUS "DEEPVAC_LIBCUDA_LIBRARIES: ${DEEPVAC_LIBCUDA_LIBRARIES}")
message(STATUS "DEEPVAC_TENSORRT_LIBRARIES: ${DEEPVAC_TENSORRT_LIBRARIES}")
message(STATUS "DEEPVAC_CV_LIBRARIES: ${DEEPVAC_CV_LIBRARIES}")

# add subdiretories
add_subdirectory(utils)
if(USE_LOADER)
    message(STATUS "Will build loader module since you enabled USE_LOADER.")
    add_subdirectory(loader)
endif()
add_subdirectory(core)
add_subdirectory(modules)

#add library
if(BUILD_STATIC)
    message(STATUS "will build libdeepvac static library")
    add_library(deepvac STATIC ${SRC_LIST})
else()
    message(STATUS "will build libdeepvac shared library")
    add_library(deepvac SHARED ${SRC_LIST})
    if(USE_STATIC_LIBTORCH)
        if(USE_TENSORRT)
            target_link_libraries(deepvac PRIVATE ${DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES} ${DEEPVAC_TENSORRT_LIBRARIES} ${DEEPVAC_LIBCUDA_LIBRARIES} ${DEEPVAC_CV_LIBRARIES})
        else()
            target_link_libraries(deepvac PRIVATE ${DEEPVAC_LIBTORCH_DEFAULT_LIBRARIES} ${DEEPVAC_LIBCUDA_LIBRARIES} ${DEEPVAC_CV_LIBRARIES})
        endif()
    endif()
endif()

target_include_directories(deepvac PUBLIC 
    "$<INSTALL_INTERFACE:include/deepvac>"   
    "$<BUILD_INTERFACE:${DEEPVAC_LIBTORCH_INCLUDE_DIRS};${HEADER_DIR_LIST};${DEEPVAC_TENSORRT_INCLUDE_DIRS};${DEEPVAC_CV_INCLUDE_DIRS}>"
)
target_compile_options(deepvac PUBLIC -fopenmp)
set_target_properties(deepvac PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(deepvac PROPERTIES LINK_FLAGS_RELEASE -s)
target_compile_options(deepvac PRIVATE -Werror)

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
