# from libdeepvac
file(GLOB LOADER_LIST src/*.cpp)
add_syszux_sources(${LOADER_LIST})
message(STATUS "found LOADER_LIST: " ${LOADER_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)