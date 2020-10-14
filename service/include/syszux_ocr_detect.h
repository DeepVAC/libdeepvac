/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <deepvac.h>
#include <torch/script.h>
#include "opencv2/opencv.hpp"
#include <queue>

namespace deepvac {
class SyszuxOcrDetect : public Deepvac {
    public:
        SyszuxOcrDetect(std::string device="cpu");
        SyszuxOcrDetect(const SyszuxOcrDetect&) = delete;
        SyszuxOcrDetect& operator=(const SyszuxOcrDetect&) = delete;
        SyszuxOcrDetect(SyszuxOcrDetect&&) = default;
        SyszuxOcrDetect& operator=(SyszuxOcrDetect&&) = default;
        virtual ~SyszuxOcrDetect() = default;
        virtual std::optional<std::vector<cv::Mat>> operator() (cv::Mat frame);
        void set(int long_size, int gap);
    private:
        void get_kernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals);
        void growing_text_line(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area);
        std::vector<std::vector<int>> adaptor_pse(torch::Tensor input_data, float min_area);
    private:
        int long_size_;
        int crop_gap_;
};
} //namespace
