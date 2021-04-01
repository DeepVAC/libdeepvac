# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

file(GLOB LOADER_SRC_LIST src/*.cpp)
add_syszux_sources(${LOADER_SRC_LIST})
message(STATUS "found LOADER_SRC_LIST: " ${LOADER_SRC_LIST})

file(GLOB HEADER_LIST include/*.h)
add_syszux_headers(${HEADER_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)