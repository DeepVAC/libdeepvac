# from libdeepvac
file(GLOB MODULE_LIST src/*.cpp)
add_syszux_sources(${MODULE_LIST})
message(STATUS "found MODULE_LIST: " ${MODULE_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)