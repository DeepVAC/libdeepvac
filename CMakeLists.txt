cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libdeepvac)

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
set(CMAKE_PREFIX_PATH /home/gemfield/libtorch/)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "TORCHLIB: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_HEADER: ${TORCH_INCLUDE_DIRS}")

#setup header files
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/loader/include)
include_directories(${CMAKE_SOURCE_DIR}/module/include)
include_directories(${CMAKE_SOURCE_DIR}/service/include)

file(GLOB UTILS_LIST ${CMAKE_SOURCE_DIR}/utils/*/*.cpp)
file(GLOB LOADER_LIST ${CMAKE_SOURCE_DIR}/loader/*/*.cpp)
file(GLOB MODULE_LIST ${CMAKE_SOURCE_DIR}/module/*/*.cpp)
file(GLOB SERVICE_LIST ${CMAKE_SOURCE_DIR}/service/*/*.cpp)
#collect source code
list(APPEND SRC_LIST ${UTILS_LIST})
list(APPEND SRC_LIST ${LOADER_LIST})
list(APPEND SRC_LIST ${MODULE_LIST})
list(APPEND SRC_LIST ${SERVICE_LIST})

message(STATUS ${SRC_LIST})

#add library
add_library(libdeepvac_static STATIC ${SRC_LIST})

#main program
include_directories(${CMAKE_SOURCE_DIR}/examples/include)
file(GLOB EXAMPLES_LIST ${CMAKE_SOURCE_DIR}/examples/*/extract_feature.cpp)
add_executable(extract_feature ${EXAMPLES_LIST})
target_link_libraries(extract_feature ${OpenCV_LIBS})
target_link_libraries(extract_feature "${TORCH_LIBRARIES}")

find_library(CRYPTO crypto)
if(CRYPTO)
    message(STATUS "found crypto library: " ${CRYPTO})
    target_link_libraries(extract_feature ${CRYPTO})
endif()

target_link_libraries(extract_feature libdeepvac_static)
set_property(TARGET extract_feature PROPERTY CXX_STANDARD 17)
