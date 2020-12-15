/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <torch/script.h>

namespace gemfield_org{
enum NORMALIZE_TYPE {
    NO_NORMALIZE,
    NORMALIZE0_1,
    NORMALIZE_1_1
};
enum MEAN_STD_TYPE {
    NO_MEAN_STD,
    MEAN_STD_FROM_IMAGENET,
    MEAN_STD_FROM_FACE
};
std::optional<cv::Mat> img2CvMat(std::string& img_path, bool is_rgb=false);
std::optional<at::Tensor> cvMat2Tensor(cv::Mat& frame, NORMALIZE_TYPE normalize=NO_NORMALIZE, MEAN_STD_TYPE mean_std=NO_MEAN_STD);
std::optional<at::Tensor> cvMat2Tensor(std::vector<cv::Mat>& frames, NORMALIZE_TYPE normalize=NO_NORMALIZE, MEAN_STD_TYPE mean_std=NO_MEAN_STD);
std::optional<at::Tensor> cvMat2Tensor(cv::Mat&& tmp_frame, NORMALIZE_TYPE normalize, MEAN_STD_TYPE mean_std);
std::optional<at::Tensor> img2Tensor(std::string& img_path, bool is_rgb=false, NORMALIZE_TYPE normalize=NO_NORMALIZE, MEAN_STD_TYPE mean_std=NO_MEAN_STD);
}
