/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_glab.h"
namespace deepvac {

float similar_min_angle = 15;
bool isSimilarAngle(float angle1, float angle2) {
    if (std::abs(angle1-angle2) < similar_min_angle or (180 - std::abs(angle1-angle2)) < similar_min_angle) {
        return true;
    }
    return false;
}

struct orderedByRatio {
    bool operator()(cv::RotatedRect const &rect1, cv::RotatedRect const &rect2) {
        float ratio1 = std::max(rect1.size.width/rect1.size.height, rect1.size.height/rect1.size.width);
        float ratio2 = std::max(rect2.size.width/rect2.size.height, rect2.size.height/rect2.size.width);
        return ratio1 > ratio2;
    }
};

struct getX { 
    bool operator()(std::vector<float> const &a, std::vector<float> const &b) const { 
        return a[0] < b[0];
    }
};

struct getY { 
    bool operator()(std::vector<float> const &a, std::vector<float> const &b) const { 
        return a[1] < b[1];
    }
};

struct getLUPlaceX {
    bool operator()(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &a, std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &b) const {
        return std::get<0>(a).most_right_xy_[0] > std::get<0>(b).most_right_xy_[0];
    }
};

struct getLUPlaceY {
    bool operator()(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &a, std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &b) const {
        return std::get<0>(a).most_bottom_xy_[1] > std::get<0>(b).most_bottom_xy_[1];
    }
};

struct getRDPlaceX {
    bool operator()(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &a, std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &b) const {
        return std::get<0>(a).most_left_xy_[0] < std::get<0>(b).most_left_xy_[0];
    }
};

struct getRDPlaceY {
    bool operator()(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &a, std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> const &b) const {
        return std::get<0>(a).most_top_xy_[1] < std::get<0>(b).most_top_xy_[1];
    }
};

AggressiveBox::AggressiveBox(cv::RotatedRect rect, std::vector<int> shape, float real_angle, bool credit_by_score, bool credit_by_shape){
    ori_rect_ = rect;
    rect_ = rect;
    init4Points();
    rect2reg_ = rect;
    shape_ = shape;
    real_angle_ = real_angle;
    credit_by_score_ = credit_by_score;
    credit_by_shape_ = credit_by_shape;
    scaleBox();
}

void AggressiveBox::init4Points() {
    cv::Mat boxes;
    cv::boxPoints(ori_rect_, boxes);
    boxes.convertTo(boxes, CV_32F);
    std::vector<std::vector<float>> boxes_vec;
    for (int i=0; i<boxes.rows; i++) {
        std::vector<float> box_vec;
        for (int j=0; j<boxes.cols; j++) {
            box_vec.push_back(boxes.at<float>(i, j));
        }
        boxes_vec.push_back(box_vec);
    }
    std::sort(boxes_vec.begin(), boxes_vec.end(), getX());
    most_left_xy_ = boxes_vec[0];
    most_right_xy_ = boxes_vec[3];
    std::sort(boxes_vec.begin(), boxes_vec.end(), getY());
    most_top_xy_ = boxes_vec[0];
    most_bottom_xy_ = boxes_vec[3];
}
void AggressiveBox::scaleBox() {
    int max_scale = std::max(shape_[0], shape_[1]);
    if (rect_.size.height >= rect_.size.width) {
        rect_.size.height += 2 * max_scale;
        scaleAxis_ = "h";
    } else {
        rect_.size.width += 2 * max_scale;
        scaleAxis_ = "w";
    }
}

cv::RotatedRect AggressiveBox::getRect() {
    return ori_rect_;
}

std::pair<float, std::vector<std::vector<cv::Point2f>>> AggressiveBox::ratio(cv::RotatedRect rect) {
    std::vector<cv::Point2f> inter_pts;
    //cv::Mat inter_pts;
    std::vector<std::vector<cv::Point2f>> contex;
    cv::rotatedRectangleIntersection(rect_, rect, inter_pts);

    if (inter_pts.size() == 0) {
        std::pair result(0.0f, contex);
        return result;
    }

    float inter_area = cv::contourArea(inter_pts);
    float inter_ratio = inter_area / (rect.size.width*rect.size.height);
    cv::Mat contex_mat;
    cv::convexHull(inter_pts, contex_mat);
    //cv::Mat contex_mat;
    contex_mat.convertTo(contex_mat, CV_32F);
    //std::vector<cv::Point2f> contex_vec;
    for (int i=0; i<contex_mat.rows; i++) {
        std::vector<cv::Point2f> contex_vec;
        cv::Point2f point;
        point.x = contex_mat.at<float>(i, 0);
        point.y = contex_mat.at<float>(i, 1);
        contex_vec.push_back(point);
        contex.push_back(contex_vec);
    }
    std::pair result(inter_ratio, contex);
    return result;
}

void AggressiveBox::addCandidateBox2Merge(AggressiveBox rect, int rect_id, float merge_ratio, std::vector<std::vector<cv::Point2f>> contex) {
    std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> candidate_box(rect, rect_id, merge_ratio, contex);
    if (real_angle_ > 45 and real_angle_ < 135) {
        if (rect.getRect().center.y < ori_rect_.center.y) {
            candidate_box_list_left_up_.push_back(candidate_box);
        } else {
            candidate_box_list_right_down_.push_back(candidate_box);
        }
    } else {
        if (rect.getRect().center.x < ori_rect_.center.x) {
            candidate_box_list_left_up_.push_back(candidate_box);
        } else {
            candidate_box_list_right_down_.push_back(candidate_box);
        }
    }
}

void AggressiveBox::sortCandidateBox() {
    if (real_angle_ > 45 and real_angle_ < 135) {
        std::sort(candidate_box_list_left_up_.begin(), candidate_box_list_left_up_.end(), getLUPlaceY());
        std::sort(candidate_box_list_right_down_.begin(), candidate_box_list_right_down_.end(), getRDPlaceY());
    } else {
        std::sort(candidate_box_list_left_up_.begin(), candidate_box_list_left_up_.end(), getLUPlaceX());
        std::sort(candidate_box_list_right_down_.begin(), candidate_box_list_right_down_.end(), getRDPlaceX());
    }
}

bool AggressiveBox::isMerge(AggressiveBox rect) {
    bool similiar = isSimilarAngle(real_angle_, rect.real_angle_);
    bool credit_by_shape = rect.credit_by_shape_;
    cv::RotatedRect rect_tmp = rect.getRect();
    auto x = ori_rect_.center.x - rect_tmp.center.x;
    auto y = ori_rect_.center.y - rect_tmp.center.y;
    float distance = std::sqrt(x*x + y*y);
    float w = (std::max(ori_rect_.size.width, ori_rect_.size.height) + std::max(rect_tmp.size.width, rect_tmp.size.height)) / 2.0f;
    float h = std::min(ori_rect_.size.width, ori_rect_.size.height) + std::min(rect_tmp.size.width, rect_tmp.size.height);
    if (distance-w > h) {
        return false;
    }

    if (not credit_by_shape) {
        return true;
    }
    float h_ratio = std::min(rect_tmp.size.width, rect_tmp.size.height) / std::min(ori_rect_.size.width, ori_rect_.size.height);
    return h_ratio>=min_h_ratio_ and h_ratio<=max_h_ratio_ and similiar;
}

void AggressiveBox::merge(std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>> rect) {
    auto contex = std::get<3>(rect);
    cv::Mat rect_box_1;
    cv::boxPoints(ori_rect_, rect_box_1);
    rect_box_1.convertTo(rect_box_1, CV_32F);
    std::vector<cv::Point> rect_box_1_vec;
    for (int i=0; i<rect_box_1.rows; i++) {
        cv::Point point;
        point.x = (int)rect_box_1.at<float>(i, 0);
        point.y = (int)rect_box_1.at<float>(i, 1);
        rect_box_1_vec.push_back(point);
    }
    if (contex.size() != 0) {
        for (int i=0; i<contex.size(); i++) {
            for (int j=0; j<contex[i].size(); j++) {
                cv::Point point_t;
                point_t.x = (int)contex[i][j].x;
                point_t.y = (int)contex[i][j].y;
                rect_box_1_vec.push_back(point_t);
            }
        }
    }
    ori_rect_ = cv::minAreaRect(rect_box_1_vec);
}

std::vector<int> AggressiveBox::mergeLeftOrUpElseRightOrBottom(std::vector<std::tuple<AggressiveBox, int, float, std::vector<std::vector<cv::Point2f>>>> candidate_box_list_left_up_or_right_down) {
    std::vector<int> result_ids;
    std::vector<int> delete_ids;
    
    while (true) {
        int merge_count = 0;
        for (int i=0; i<candidate_box_list_left_up_or_right_down.size(); i++) {
            auto rect = candidate_box_list_left_up_or_right_down[i];
            bool flag = false;
            for (auto id : delete_ids) {
                if (id == i) {
                    flag = true;
                }
            }
            if (flag) {
                continue;
            }
            if (!isMerge(std::get<0>(rect))) {
                return result_ids;
            }
            merge(rect);
            delete_ids.push_back(i);
            result_ids.push_back(std::get<1>(rect));
            merge_count += 1;
        }
        if (merge_count == 0) {
            break;
        }
    }
    return result_ids;
}

std::vector<int> AggressiveBox::mergeRects() {
    std::vector<int> merge_ids;
    std::vector<int> LU_ids = mergeLeftOrUpElseRightOrBottom(candidate_box_list_left_up_);
    std::vector<int> RD_ids = mergeLeftOrUpElseRightOrBottom(candidate_box_list_right_down_);
    merge_ids.insert(merge_ids.end(), LU_ids.begin(), LU_ids.end());
    merge_ids.insert(merge_ids.end(), RD_ids.begin(), RD_ids.end());
    return merge_ids;
}

DeepvacOcrFrame::DeepvacOcrFrame(cv::Mat img, std::vector<cv::RotatedRect> rect_list, bool is_oneway) {
    shape_ = {img.rows, img.cols};
    rect_list_ = rect_list;
    is_oneway_ = is_oneway;
    sortBoxByRatio();
    initDominantAngle();
    for (auto rect : rect_list_) {
        aggressive_box_list_.push_back(creatAggressiveBox(rect));
    }
}

void DeepvacOcrFrame::sortBoxByRatio() {
    std::sort(rect_list_.begin(), rect_list_.end(), orderedByRatio());
}

int DeepvacOcrFrame::initDominantAngle() {
    for (auto rect : rect_list_) {
        float real_angle;
        if (rect.size.width < rect.size.height) {
            real_angle = std::abs(rect.angle - 90);
        } else {
            real_angle = std::abs(rect.angle);
        }
        if (real_angle == 180) {
            real_angle = 0;
        }
        real_angle_list_.push_back(real_angle);
    }
    total_box_num_ = real_angle_list_.size();
    auto sort_angle_list = real_angle_list_;
    std::sort(sort_angle_list.begin(), sort_angle_list.end());
    int size = sort_angle_list.size();
    if (size % 2 == 0) {
        median_angle_ = (sort_angle_list[size/2-1] + sort_angle_list[size/2]) / 2;
    } else {
        median_angle_ = sort_angle_list[(size-1)/2];
    }
    similar_box_num_ = 0;
    for (auto angle: real_angle_list_) {
        if (isSimilarAngle(angle, median_angle)) {
            similar_box_num_ += 1;
        }
    }
    if (is_oneway_) {
        return 0;
    }
    if (similar_box_num_ == total_box_num_) {
        is_oneway_ = true;
        return 0;
    }

    float similar_box_ratio = similar_box_num_ * 1.0f / total_box_num_;
    if (similar_box_ratio > similar_box_ratio_) {
        is_oneway_ = true;
        return 0;
    }
    return 0;
}

AggressiveBox DeepvacOcrFrame::creatAggressiveBox(cv::RotatedRect rect) {
    float real_angle;
    if (rect.size.width < rect.size.height) {
        real_angle = std::abs(rect.angle - 90);
    } else {
        real_angle = std::abs(rect.angle);
    }
    if (real_angle == 180) {
        real_angle = 0;
    }

    bool credit_by_shape = false;
    if (std::max(rect.size.width, rect.size.height)/std::min(rect.size.width, rect.size.height) >= credit_shape_ratio_) {
        credit_by_shape = true;
    }
    return AggressiveBox(rect, shape_, real_angle, true, credit_by_shape);
}

void DeepvacOcrFrame::aggressive4mergePeer(AggressiveBox& aggressive_rect, int offset) {
    for (int i=offset+1; i<aggressive_box_list_.size(); i++) {
        auto rect = aggressive_box_list_[i];
        if (!rect.valid_) {
            continue;
        }
        auto res_pair = aggressive_rect.ratio(rect.getRect());
        auto merge_ratio = res_pair.first;
        auto convex = res_pair.second;
        if (merge_ratio > merge_ratio_) {
            aggressive_rect.addCandidateBox2Merge(rect, i, merge_ratio, convex);
        }
    }
    aggressive_rect.sortCandidateBox();
    std::vector<int> merge_ids = aggressive_rect.mergeRects();

    for (auto merge_id : merge_ids) {
        aggressive_box_list_[merge_id].valid_ = false;
    }
}

std::optional<std::vector<AggressiveBox>> DeepvacOcrFrame::operator() (){
    if (aggressive_box_list_.size() == 0) {
        return std::nullopt;
    }
    std::vector<AggressiveBox> new_rect_list;
    for (int i=0; i<aggressive_box_list_.size(); i++) {
        auto rect = aggressive_box_list_[i];
        if (!rect.valid_) {
            continue;
        }
        aggressive4mergePeer(rect, i);
        new_rect_list.push_back(rect);
    }
    return new_rect_list;
}

}//namespace
