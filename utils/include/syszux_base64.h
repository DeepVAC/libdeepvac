#pragma once

#include <string>
#include <vector>

namespace gemfield_org {
std::vector<uint8_t> unbase64(const std::string& base64_data);
std::string base64(const unsigned char* bin, size_t len, int lineLenght = -1);

}// namespace
