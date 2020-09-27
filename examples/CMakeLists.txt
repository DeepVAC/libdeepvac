# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

file(GLOB EXAMPLES_LIST src/*.cpp)
message(STATUS "found EXAMPLES_LIST: " ${EXAMPLES_LIST})

#main program
file(GLOB SYSZUX_EXTRACT */syszux_extract_feature.cpp)

find_library(CRYPTO crypto)
if(CRYPTO)
    message(STATUS "found crypto library: " ${CRYPTO})
endif()

#test examples
set(examples_dest "bin")

file( GLOB EXAMPLES_LIST ${CMAKE_SOURCE_DIR}/examples/*/test_*.cpp )
foreach( testsyszuxfile ${EXAMPLES_LIST} )
    get_filename_component(testfile "${testsyszuxfile}" NAME)
    message(STATUS "Add test binary: ${testfile}")
    string(REPLACE ".cpp" "" testname ${testfile} )
    add_executable( ${testname} ${testsyszuxfile} )
    target_link_libraries( ${testname} deepvac ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    target_include_directories(${testname} PUBLIC 
        "$<INSTALL_INTERFACE:include/deepvac>"   
        "$<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    install(TARGETS ${testname} DESTINATION "${examples_dest}")
endforeach( testsyszuxfile ${EXAMPLES_LIST} )
