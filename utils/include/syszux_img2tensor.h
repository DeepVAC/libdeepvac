/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/script.h>

namespace gemfield_org{
std::optional<cv::Mat> img2CvMat(std::string& img_path, bool is_rgb=false);
std::optional<at::Tensor> cvMat2Tensor(cv::Mat& frame, bool is_normalize=true);
std::optional<at::Tensor> img2Tensor(std::string& img_path, bool is_rgb=false, bool is_normalize=true);
}