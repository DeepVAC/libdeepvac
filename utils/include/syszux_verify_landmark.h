/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "opencv2/opencv.hpp"
#include <torch/script.h>
#include <vector>

namespace gemfield_org{
    bool isValidLandmark(std::vector<float> bbox, std::vector<float> landmark, int min_face_size=24);
    torch::Tensor getDecodeBox(torch::Tensor& prior, torch::Tensor& variances, torch::Tensor& loc);
    void decodeLandmark(torch::Tensor& prior, torch::Tensor& variances, torch::Tensor& landms);
}
