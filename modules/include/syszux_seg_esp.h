/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include "deepvac.h"
#include "syszux_img2tensor.h"

namespace deepvac{
class SyszuxSegEsp : public Deepvac{
    public:
        using Deepvac::Deepvac;
        SyszuxSegEsp(const SyszuxSegEsp&) = default;
        SyszuxSegEsp& operator=(const SyszuxSegEsp&) = default;
        SyszuxSegEsp(SyszuxSegEsp&&) = default;
        SyszuxSegEsp& operator=(SyszuxSegEsp&&) = default;
        virtual ~SyszuxSegEsp() = default;
    public:
        void set(std::vector<int> image_size);
        std::optional<cv::Mat> process(cv::Mat frame);	
    private:
        std::vector<int> image_size_;
};
} //namespace deepvac
