/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include <vector>
#include <tuple>
#include "opencv2/opencv.hpp"

namespace gemfield_org{
class AlignFace
{
    public:
        AlignFace();
        ~AlignFace() = default;
        std::tuple<cv::Mat, std::vector<float>> operator() (cv::Mat& frame, cv::Mat& facial_5pts);
    private:
        cv::Mat warpAndCrop(cv::Mat& src_img, cv::Mat& facial_5pts);
        cv::Mat getAffineTransform(cv::Mat& facial_5pts);
        cv::Mat findNonereflectiveSimilarity(cv::Mat& facial_5pts, cv::Mat& ref_facial_5pts);
        cv::Mat tformfwd(cv::Mat& trans, cv::Mat facial_5pts);
        std::vector<float> pointsTransform(const std::vector<cv::Point2f>& points, const cv::Mat& matrix);
    private:
        cv::Mat ref_facial_5pts_;
        cv::Mat crop_size_;
};
}//namespace gemfield_org
