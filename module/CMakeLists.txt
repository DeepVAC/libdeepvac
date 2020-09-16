# from libdeepvac
file(GLOB MODULE_LIST src/*.cpp)
add_syszux_sources(${MODULE_LIST})
message(STATUS "found MODULE_LIST: " ${MODULE_LIST})

file(GLOB HEADER_LIST include/*.h)
add_syszux_headers(${HEADER_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)