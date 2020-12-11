/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "opencv2/opencv.hpp"
#include "deepvac.h"

namespace deepvac {
class SyszuxClsMobile : public Deepvac {
    public:
        SyszuxClsMobile(std::string path, std::string device="cpu");
        SyszuxClsMobile(const SyszuxClsMobile&) = delete;
        SyszuxClsMobile& operator=(const SyszuxClsMobile&) = delete;
        SyszuxClsMobile(SyszuxClsMobile&&) = default;
        SyszuxClsMobile& operator=(SyszuxClsMobile&&) = default;
        virtual ~SyszuxClsMobile() = default;
    public:
        std::optional<std::vector<cv::Mat>> process (std::vector<cv::Mat> crop_imgs);
        void set(float confidence, float threshold);
    private:
        float getScore(std::vector<float> confidences);
        std::vector<cv::Mat> getRotatedImgs(std::vector<cv::Mat> crop_imgs, std::vector<float> confidences);
    private:
        float confidence_;
        float threshold_;
};
} //namespace deepvac
