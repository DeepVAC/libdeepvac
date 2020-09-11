/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <torch/script.h>

namespace gemfield_org{
    torch::Tensor nms(torch::Tensor& dets, float threshold);
}