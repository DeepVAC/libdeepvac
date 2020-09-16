# from libdeepvac
file(GLOB LOADER_LIST src/*.cpp)
add_syszux_sources(${LOADER_LIST})
message(STATUS "found LOADER_LIST: " ${LOADER_LIST})

file(GLOB HEADER_LIST include/*.h)
add_syszux_headers(${HEADER_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)