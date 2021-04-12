# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

file(GLOB EXAMPLES_LIST src/*.cpp)
macro (add_syszux_flags)
    foreach (_src ${ARGN})
        list (APPEND FLAGS_FILE_LIST "${_src}")
    endforeach()
    set (FLAGS_FILE_LIST ${FLAGS_FILE_LIST} PARENT_SCOPE)
endmacro()

find_library(CRYPTO crypto)
if(CRYPTO)
    message(STATUS "found crypto library: " ${CRYPTO})
endif()

#test examples
set(examples_dest "bin")

file( GLOB EXAMPLES_LIST ${CMAKE_SOURCE_DIR}/examples/*/test_*.cpp )
file( GLOB EXAMPLES_LIST_MIN ${CMAKE_SOURCE_DIR}/examples/*/test_*retina*.cpp)
file( GLOB EXAMPLES_BENCHMARK ${CMAKE_SOURCE_DIR}/examples/*/test_*benchmark.cpp)
list(APPEND EXAMPLES_LIST_MIN ${EXAMPLES_BENCHMARK})
if(NOT USE_TENSORRT)
    list(FILTER EXAMPLES_LIST EXCLUDE REGEX ".*_nv.cpp$")
    list(FILTER EXAMPLES_LIST_MIN EXCLUDE REGEX ".*_nv.cpp$")
endif()
#internal test
add_syszux_flags(face_retina_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_face_retina.cpp)
add_syszux_flags(ocr_pse_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_ocr_pse.cpp)
add_syszux_flags(seg_esp_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_seg_esp.cpp)
add_syszux_flags(cls_mobile_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_cls_mobile.cpp)
add_syszux_flags(detect_yolo_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_detect_yolo.cpp)
add_syszux_flags(cls_resnet_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_cls_resnet.cpp)

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

if(BUILD_ALL_EXAMPLES)
    set(EXAMPLES_LIST_MIN ${EXAMPLES_LIST})
endif()

if(NOT USE_TENSORRT)
    list(FILTER EXAMPLES_LIST_MIN EXCLUDE REGEX ".*_nv.cpp$")
endif()

message(STATUS "found EXAMPLES_SRC_LIST: " ${EXAMPLES_LIST_MIN})

foreach( testsyszuxfile ${EXAMPLES_LIST_MIN} )
    get_filename_component(testfile "${testsyszuxfile}" NAME)
    message(STATUS "Add test binary: ${testfile}")
    string(REPLACE ".cpp" "" testname ${testfile} )
    add_executable( ${testname} ${testsyszuxfile} )

    if(USE_STATIC_LIBTORCH)
        target_compile_options(${testname} PRIVATE -fopenmp)
        target_link_options(${testname} PRIVATE -fopenmp)
        if(BUILD_STATIC)
            if(USE_CUDA)
                if(USE_TENSORRT)
                    target_link_libraries( ${testname} ${DEEPVAC_LIBRARIES} ${DEEPVAC_CV_LIBRARIES} ${DEEPVAC_LIBTORCH_CUDA_LIBRARIES} ${DEEPVAC_LIBCUDA_LIBRARIES} ${DEEPVAC_TENSORRT_LIBRARIES})
                else()
                    target_link_libraries( ${testname} ${DEEPVAC_LIBRARIES} ${DEEPVAC_CV_LIBRARIES} ${DEEPVAC_LIBTORCH_CUDA_LIBRARIES} ${DEEPVAC_LIBCUDA_LIBRARIES})
                endif()
            else()
                target_link_libraries( ${testname} ${DEEPVAC_LIBRARIES} ${DEEPVAC_CV_LIBRARIES} ${DEEPVAC_LIBTORCH_CPU_LIBRARIES})
            endif()
        else()
            target_link_libraries( ${testname} ${DEEPVAC_LIBRARIES} ${DEEPVAC_CV_LIBRARIES} -lrt)
        endif()
    else()
        target_link_libraries( ${testname} ${DEEPVAC_LIBRARIES} ${DEEPVAC_CV_LIBRARIES} ${TORCH_LIBRARIES})
    endif()

    target_include_directories(${testname} PUBLIC
        "$<INSTALL_INTERFACE:include/deepvac>"   
        "$<BUILD_INTERFACE:${DEEPVAC_LIBTORCH_INCLUDE_DIRS};${DEEPVAC_TENSORRT_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    #install(TARGETS ${testname} DESTINATION "${examples_dest}")
endforeach( testsyszuxfile ${EXAMPLES_LIST_MIN} )