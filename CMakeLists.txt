cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libdeepvac)

# define library version
set(libdeepvac_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(libdeepvac_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(libdeepvac_VERSION ${libdeepvac_VERSION_MAJOR}.${libdeepvac_VERSION_MINOR} CACHE STRING "version" FORCE)

# some options
option(libdeepvac_DEBUG_MODE "whether or not debug mode is activated" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

#setup library output dir
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#setup binary output dir
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

#setup local
#add_definitions(-DGEMFIELD)

#replace with your libtorch dir
#set(CMAKE_PREFIX_PATH /home/gemfield/libtorch/)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

macro (add_syszux_sources)
    file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        message(STATUS "found src: " ${_src})
        list (APPEND SRC_LIST "${_src}")
    endforeach()
    if (_relPath)
        set (SRC_LIST ${SRC_LIST} PARENT_SCOPE)
    endif()
endmacro()

macro (add_header_dir)
    foreach (_header_dir ${ARGN})
        message(STATUS "Add header include: " ${_header_dir})
        list (APPEND HEADER_DIR_LIST "${_header_dir}")
    endforeach()
    set (HEADER_DIR_LIST ${HEADER_DIR_LIST} PARENT_SCOPE)
endmacro()

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "TORCHLIB: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_HEADER: ${TORCH_INCLUDE_DIRS}")

# add subdiretories
add_subdirectory(utils)
add_subdirectory(loader)
add_subdirectory(module)
add_subdirectory(service)

#setup header files
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "HEADER_DIR_LIST: " ${HEADER_DIR_LIST})
include_directories(${HEADER_DIR_LIST})

message(STATUS "SRC_LIST: " ${SRC_LIST})

#add library
add_library(libdeepvac_static STATIC ${SRC_LIST})

#add examples
add_subdirectory(examples)


