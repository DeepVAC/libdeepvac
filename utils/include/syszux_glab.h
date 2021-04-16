/*
 *  * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 *   * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 *    * You may not use this file except in compliance with the License.
 *     */

#pragma once
// #include "syszux_ocr_pse.h"
#include "opencv2/opencv.hpp"

namespace deepvac {

class AggressiveBox {
    public:
        AggressiveBox(cv::RotatedRect rect, std::vector<int> shape, float real_angle, bool credit_by_score=true, bool credit_by_shape=true);
        void init4Points();
        void scaleBox();
        cv::RotatedRect getRect();
        std::pair<float, std::vector<std::vector<cv::Point2f>>> ratio(cv::RotatedRect rect);
        void addCandidateBox2Merge(AggressiveBox rect, int rect_id, float merge_ratio, std::vector<std::vector<cv::Point2f>> contex);
        void sortCandidateBox();
        bool isMerge(AggressiveBox rect);
        void merge(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> rect);
        std::vector<int> mergeLeftOrUpElseRightOrBottom(std::vector<std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>>> candidate_box_list_left_up_or_right_down);
        std::vector<int> mergeRects();
    private:
        float min_h_ratio_ = 0.75;
        float max_h_ratio_ = 1.3;
        float box_min_wh_ratio_ = 2.0;

        cv::RotatedRect rect_;
        cv::RotatedRect ori_rect_;
        cv::RotatedRect rect2reg_;

        std::vector<int> shape_;
        bool credit_by_score_;
        bool credit_by_shape_;
        std::vector<std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>>> candidate_box_list_left_up_;
        std::vector<std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>>> candidate_box_list_right_down_;
        float real_angle_;
        std::string scaleAxis_;
    public:
        std::vector<float> most_left_xy_;
        std::vector<float> most_right_xy_;
        std::vector<float> most_top_xy_;
        std::vector<float> most_bottom_xy_;
        bool valid_ = true;
};

class DeepvacOcrFrame {
    public:
        DeepvacOcrFrame(cv::Mat img, std::vector<cv::RotatedRect> rect_list, bool is_oneway=false);
        void sortBoxByRatio();
        int initDominantAngle();
        AggressiveBox creatAggressiveBox(cv::RotatedRect rect);
        void aggressive4mergePeer(AggressiveBox& aggressive_rect, int offset);
        std::optional<std::vector<AggressiveBox>> operator() ();
    private:
        float merge_ratio_ = 0.7;
        float similar_box_ratio_ = 0.95;
        float credit_shape_ratio_ = 2.0;
        std::vector<int> shape_;
        float median_angle = 0;
        std::vector<cv::RotatedRect> rect_list_;
        bool is_oneway_;
        std::vector<AggressiveBox> aggressive_box_list_;
        std::vector<float> real_angle_list_;
        int total_box_num_;
        int similar_box_num_;
        float median_angle_;
};

} //namespace deepvac
