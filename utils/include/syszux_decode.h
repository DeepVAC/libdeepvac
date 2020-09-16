/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <torch/script.h>

namespace gemfield_org{
    int decodeBox(torch::Tensor loc, torch::Tensor prior, torch::Tensor variances, torch::Tensor& boxes);
    int decodeLandmark(torch::Tensor prior, torch::Tensor variances, torch::Tensor& landms);
}
