/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <torch/script.h>
#include "opencv2/opencv.hpp"
#include <queue>

//namespace gemfield_org {
class SyszuxOcrDetect{
    public:
        SyszuxOcrDetect(std::string device);
        std::optional<cv::Mat> operator() (cv::Mat img, int long_size);
    private:
        void get_kernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals);
        void growing_text_line(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area);
        std::vector<std::vector<int>> adaptor_pse(torch::Tensor input_data, float min_area);
    private:
        std::string device_;
};
//} //namespace
