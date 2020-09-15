/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "opencv2/opencv.hpp"
#include <vector>

namespace gemfield_org{
    bool filterFace(cv::Mat img, std::vector<float> bbox, std::vector<float> landmark);
}
