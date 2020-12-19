/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "opencv2/opencv.hpp"
#include "deepvac.h"
#include "syszux_img2tensor.h"
#include "syszux_nms.h"

namespace deepvac {
class SyszuxDetectYolo : public Deepvac {
    public:
        SyszuxDetectYolo(std::string path, std::string device="cpu");
        SyszuxDetectYolo(std::vector<unsigned char>&& buffer, std::string device = "cpu");
        SyszuxDetectYolo(const SyszuxDetectYolo&) = delete;
        SyszuxDetectYolo& operator=(const SyszuxDetectYolo&) = delete;
        SyszuxDetectYolo(SyszuxDetectYolo&&) = default;
        SyszuxDetectYolo& operator=(SyszuxDetectYolo&&) = default;
        virtual ~SyszuxDetectYolo() = default;
    public:
        void set(int input_size, float iou_thresh, float score_thresh);
        std::optional<std::vector<std::pair<int, std::vector<float>>>> process (cv::Mat& frame);
    private:
        torch::Tensor postProcess(torch::Tensor& preds);
    private:
        int input_size_;
        float iou_thresh_;
        float score_thresh_;
        std::vector<std::string> idx_to_cls_;
};
} //namespace deepvac
