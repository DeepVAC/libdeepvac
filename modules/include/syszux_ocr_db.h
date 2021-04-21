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
#include "syszux_glab.h"
#include "syszux_clipper.h"
#include <queue>
#include <algorithm>

namespace deepvac {
class SyszuxOcrDB : public Deepvac {
    public:
        using Deepvac::Deepvac;
        SyszuxOcrDB(const SyszuxOcrDB&) = default;
        SyszuxOcrDB& operator=(const SyszuxOcrDB&) = default;
        SyszuxOcrDB(SyszuxOcrDB&&) = default;
        SyszuxOcrDB& operator=(SyszuxOcrDB&&) = default;
        virtual ~SyszuxOcrDB() = default;
    public:
        std::optional<std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>>> process(cv::Mat frame);
        void set(int long_size, int crop_gap, int text_min_area, int text_min_size, float text_mean_score, float text_thresh, float unclip_ratio);
        void setGlab(bool is_glab);
	void setExtend(bool is_extend);
        void setUnclip(bool is_unclip);
        void setPolygonScore(bool is_polygon_score);
    private:
        cv::Mat cropRect(cv::Mat &img, cv::RotatedRect &rotated_rects);
        float boxScoreFast(cv::Mat &pred, cv::RotatedRect &rect);
        float polygonScoreAcc(cv::Mat &pred, std::vector<cv::Point> &contour);
        std::optional<cv::RotatedRect> unClip(cv::RotatedRect &rect);
        void getContourArea(const std::vector<std::vector<float>> &box, float &distance);
    private:
        int long_size_{1280};
        int crop_gap_{10};
        int text_min_area_{300};
        int text_min_size_{3};
        float text_mean_score_{0.5};
        float text_thresh_{0.3};
        float unclip_ratio_{1.5};
        bool is_glab_{false};
        bool is_extend_{false};
        bool is_unclip_{false};
        bool is_polygon_score_{false};
};
} //namespace
