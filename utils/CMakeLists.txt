# from libdeepvac
file(GLOB UTILS_LIST src/*.cpp)
add_syszux_sources(${UTILS_LIST})
message(STATUS "found UTILS_LIST: " ${UTILS_LIST})

add_header_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)