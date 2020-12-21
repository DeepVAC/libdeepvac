/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "opencv2/opencv.hpp"
#include "deepvac.h"
#include "syszux_img2tensor.h"

namespace deepvac {
class SyszuxClsMobile : public Deepvac {
    public:
        using Deepvac::Deepvac;
        SyszuxClsMobile(const SyszuxClsMobile&) = default;
        SyszuxClsMobile& operator=(const SyszuxClsMobile&) = default;
        SyszuxClsMobile(SyszuxClsMobile&&) = default;
        SyszuxClsMobile& operator=(SyszuxClsMobile&&) = default;
        virtual ~SyszuxClsMobile() = default;
    public:
        std::optional<std::pair<int, float>> process (cv::Mat frame);
        std::optional<std::vector<std::pair<int, float>>> process(std::vector<cv::Mat> frame);
};
} //namespace deepvac
