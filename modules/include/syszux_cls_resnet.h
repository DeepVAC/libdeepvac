/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include "deepvac.h"
#include "syszux_img2tensor.h"

namespace deepvac{
class SyszuxClsResnet : public Deepvac{
    public:
        using Deepvac::Deepvac;
        SyszuxClsResnet(const SyszuxClsResnet&) = default;
        SyszuxClsResnet& operator=(const SyszuxClsResnet&) = default;
        SyszuxClsResnet(SyszuxClsResnet&&) = default;
        SyszuxClsResnet& operator=(SyszuxClsResnet&&) = default;
        virtual ~SyszuxClsResnet() = default;
        std::optional<std::pair<int, float>> process(cv::Mat frame);
        void set(std::vector<int> input_size);
    private:
        std::vector<int> input_size_;
};
} //namespace deepvac
