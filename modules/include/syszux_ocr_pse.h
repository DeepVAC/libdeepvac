/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <deepvac.h>
#include <torch/script.h>
#include "opencv2/opencv.hpp"
#include "syszux_img2tensor.h"
#include <queue>
#include <algorithm>

namespace deepvac {
class SyszuxOcrPse : public Deepvac {
    public:
        using Deepvac::Deepvac;
        SyszuxOcrPse(const SyszuxOcrPse&) = default;
        SyszuxOcrPse& operator=(const SyszuxOcrPse&) = default;
        SyszuxOcrPse(SyszuxOcrPse&&) = default;
        SyszuxOcrPse& operator=(SyszuxOcrPse&&) = default;
        virtual ~SyszuxOcrPse() = default;
    public:
        std::optional<std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>>> process(cv::Mat frame);
        void set(int long_size, int gap);
    private:
        void getKernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals);
        void growingTextLine(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area);
        std::vector<std::vector<int>> adaptorPse(torch::Tensor input_data, float min_area);
        bool isMerge(std::vector<float> rect1, std::vector<float> rect2);
        std::vector<std::vector<float>> mergeBox(std::vector<std::vector<float>> rects);
        cv::Mat cropRect(cv::Mat &img, cv::RotatedRect &rotated_rects);
    private:
        int long_size_{1280};
        int crop_gap_{10};
};
} //namespace
