# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

file(GLOB EXAMPLES_LIST src/*.cpp)
message(STATUS "found EXAMPLES_LIST: " ${EXAMPLES_LIST})

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
#internal test
add_syszux_flags(face_retina_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_face_retina.cpp)
add_syszux_flags(ocr_pse_deepvac@${CMAKE_CURRENT_SOURCE_DIR}/src/test_ocr_pse.cpp)

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

foreach( testsyszuxfile ${EXAMPLES_LIST} )
    get_filename_component(testfile "${testsyszuxfile}" NAME)
    message(STATUS "Add test binary: ${testfile}")
    string(REPLACE ".cpp" "" testname ${testfile} )
    add_executable( ${testname} ${testsyszuxfile} )

    if(USE_STATIC_LIBTORCH)
        target_compile_options(${testname} PRIVATE -fopenmp)
        target_link_options(${testname} PRIVATE -fopenmp)
        target_link_libraries( ${testname} deepvac ${TORCH_STATIC_LIBRARIES} ${CUDA_LIBRARIES_ALL} ${OpenCV_LIBS})
    else()
        target_link_libraries( ${testname} deepvac ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    endif()
    target_include_directories(${testname} PUBLIC 
        "$<INSTALL_INTERFACE:include/deepvac>"   
        "$<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    install(TARGETS ${testname} DESTINATION "${examples_dest}")
endforeach( testsyszuxfile ${EXAMPLES_LIST} )