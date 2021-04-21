/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "syszux_ocr_db.h"

namespace deepvac {

void SyszuxOcrDB::set(int long_size, int crop_gap, int text_min_area, int text_min_size, float text_mean_score, float text_thresh, float unclip_ratio){
    long_size_ = long_size;
    crop_gap_ = crop_gap;
    text_min_area_ = text_min_area;
    text_min_size_ = text_min_size;
    text_mean_score_ = text_mean_score;
    text_thresh_ = text_thresh;
    unclip_ratio_ = unclip_ratio;
}

void SyszuxOcrDB::setGlab(bool is_glab){
    is_glab_ = is_glab;
}

void SyszuxOcrDB::setExtend(bool is_extend){
    is_extend_ = is_extend;
}

void SyszuxOcrDB::setUnclip(bool is_unclip){
    is_unclip_ = is_unclip;
}

void SyszuxOcrDB::setPolygonScore(bool is_polygon_score){
    is_polygon_score_ = is_polygon_score;
}

cv::Mat SyszuxOcrDB::cropRect(cv::Mat &img, cv::RotatedRect &rotated_rects) {
    cv::Point2f center = rotated_rects.center;
    cv::Size2f size = rotated_rects.size;
    float angle = rotated_rects.angle;
    cv::Point center_i;
    cv::Size size_i;
    center_i.x = int(center.x);
    center_i.y = int(center.y);
    size_i.width = int(size.width);
    size_i.height = int(size.height);

    if (size_i.width < size_i.height) {
        angle += 90.;
        int temp = size_i.width;
        size_i.width = size_i.height;
        size_i.height = temp;
    }
    auto M = cv::getRotationMatrix2D(center_i, angle, 1);
    cv::Mat img_rot, img_crop;
    cv::warpAffine(img, img_rot, M, img.size(), cv::INTER_CUBIC);
    cv::getRectSubPix(img_rot, size_i, center_i, img_crop);
    return img_crop;
}

float SyszuxOcrDB::polygonScoreAcc(cv::Mat &pred, std::vector<cv::Point> &contour){
    int width = pred.cols;
    int height = pred.rows;
    std::vector<float> box_x;
    std::vector<float> box_y;
    for(int i=0; i<contour.size(); ++i){
        box_x.push_back(contour[i].x);
        box_y.push_back(contour[i].y);
    }

    int xmin = std::clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0, width - 1);
    int xmax = std::clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0, width - 1);
    int ymin = std::clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0, height - 1);
    int ymax = std::clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0, height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point rook_point[contour.size()];
    for(int i=0; i<contour.size(); ++i){
        rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
    }
    const cv::Point *ppt[1] = {rook_point};
    int npt[] = {int(contour.size())};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(croppedImg);
    float score = cv::mean(croppedImg, mask)[0];
    return score;
}

float SyszuxOcrDB::boxScoreFast(cv::Mat &pred, cv::RotatedRect &rect){
    int width = pred.cols;
    int height = pred.rows;
    cv::Mat vertex;
    cv::boxPoints(rect, vertex);
    if(vertex.rows!=4){
        std::cout<<"[GEMFIELD] warning: cv::boxPoints return vertex length != 4"<<std::endl;
        return 0.0;
    }
    float box_x[4] = {vertex.at<float>(0, 0), vertex.at<float>(1, 0), vertex.at<float>(2, 0), vertex.at<float>(3, 0)};
    float box_y[4] = {vertex.at<float>(0, 1), vertex.at<float>(1, 1), vertex.at<float>(2, 1), vertex.at<float>(3, 1)};

    int xmin = std::clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0, width - 1);
    int xmax = std::clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0, width - 1);
    int ymin = std::clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0, height - 1);
    int ymax = std::clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0, height - 1);

    cv::Mat mask;
    mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

    cv::Point rook_point[4];
    rook_point[0] = cv::Point(int(box_x[0]) - xmin, int(box_y[0]) - ymin);
    rook_point[1] = cv::Point(int(box_x[1]) - xmin, int(box_y[1]) - ymin);
    rook_point[2] = cv::Point(int(box_x[2]) - xmin, int(box_y[2]) - ymin);
    rook_point[3] = cv::Point(int(box_x[3]) - xmin, int(box_y[3]) - ymin);
    const cv::Point *ppt[1] = {rook_point};
    int npt[] = {4};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImg;
    pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(croppedImg);
    float score = cv::mean(croppedImg, mask)[0];
    return score;
}

void SyszuxOcrDB::getContourArea(const std::vector<std::vector<float>> &box, float &distance){
    int pts_num = 4;
    float area = 0.0f;
    float dist = 0.0f;
    for (int i = 0; i < pts_num; i++) {
        area += box[i][0] * box[(i + 1) % pts_num][1] - box[i][1] * box[(i + 1) % pts_num][0];
	dist += sqrtf(std::pow(box[i][0]-box[(i + 1)%pts_num][0], 2) + std::pow(box[i][1] - box[(i + 1) % pts_num][1], 2));
    }
    area = fabs(float(area / 2.0));
    distance = area * unclip_ratio_ / dist;
}

std::optional<cv::RotatedRect> SyszuxOcrDB::unClip(cv::RotatedRect &rect){
    cv::Mat vertex;
    cv::boxPoints(rect, vertex);

    std::vector<std::vector<float>> box;
    for (int i = 0; i < vertex.rows; ++i) {
        std::vector<float> tmp;
        for (int j = 0; j < vertex.cols; ++j) {
            tmp.push_back(vertex.at<float>(i, j));
        }
        box.push_back(tmp);
    }

    float distance = 1.0;
    getContourArea(box, distance);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
      << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
      << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
      << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;

    for (int j = 0; j < soln.size(); j++) {
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    if (points.size() <= 0){
        return std::nullopt;
    }
    cv::RotatedRect res;
    res = cv::minAreaRect(points);
    return res;
}

std::optional< std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> > SyszuxOcrDB::process(cv::Mat img){
    GEMFIELD_SI;
    //prepare input
    std::vector<cv::Mat> crop_imgs;
    std::vector<std::vector<int>> rects;
    cv::Mat img_ori = img.clone();
    cv::Mat resize_img, rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    float scale1 = long_size_ * 1.0 / std::max(img.rows, img.cols);
    cv::resize(rgb_img, resize_img, cv::Size(), scale1, scale1);
    
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(resize_img, gemfield_org::NORMALIZE0_1, gemfield_org::MEAN_STD_FROM_IMAGENET, device_);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();

    // forward
    auto outputs = forward(input_tensor);

    // binarize probability map
    outputs = outputs.squeeze();
    torch::Tensor pred = outputs.select(0, 0).to(torch::kCPU).toType(torch::kFloat);
    torch::Tensor segmentation = (pred>text_thresh_);
    segmentation = segmentation.mul(255).clamp(0, 255).toType(torch::kU8);

    // binary map and pre convert to cvMat
    cv::Mat image_binary(segmentation.size(0), segmentation.size(1), CV_8UC1);
    std::memcpy((void*)image_binary.data, segmentation.data_ptr(), torch::elementSize(torch::kU8) * segmentation.numel());
    cv::Mat pred_Mat(pred.size(0), pred.size(1), CV_32FC1);
    std::memcpy((void*)pred_Mat.data, pred.data_ptr(), torch::elementSize(torch::kFloat) * pred.numel());

    // binary cvMat find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image_binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // extract text box to db_detect_out
    std::vector<float> scale2 = {(float)(img.cols * 1.0 / pred.size(1)), (float)(img.rows * 1.0 / pred.size(0))};
    std::vector<cv::RotatedRect> db_detect_out;
    for(int i=0; i<contours.size(); ++i){
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        if(std::max(rect.size.width, rect.size.width) < text_min_size_)
            continue;
        if(rect.size.width * rect.size.width < text_min_area_)
            continue;
        float score;
        if(is_polygon_score_){
            score = polygonScoreAcc(pred_Mat, contours[i]);
        } else{
            score = boxScoreFast(pred_Mat, rect);
        }
        if(score < text_mean_score_)
            continue;
        rect.center.x = rect.center.x * scale2[0];
        rect.center.y = rect.center.y * scale2[1];
        rect.size.width  = rect.size.width * scale2[0];
        rect.size.height = rect.size.height * scale2[1];

        // dilated text box
        if(is_unclip_){
            auto unclip_rect = unClip(rect);
            if(unclip_rect) db_detect_out.push_back(unclip_rect.value());
            continue;
        }
        if(rect.size.width >= rect.size.height){
            rect.size.width = rect.size.width + rect.size.height * (unclip_ratio_ - 1);
            rect.size.height = rect.size.height * unclip_ratio_;
        } else {
            rect.size.height = rect.size.height + rect.size.width * (unclip_ratio_ - 1);
            rect.size.width = rect.size.width * unclip_ratio_;
        }
        db_detect_out.push_back(rect);
    }

    if(db_detect_out.size()<=0)
        return std::nullopt;

    std::vector<cv::RotatedRect> result;
    if(is_glab_){
        deepvac::DeepvacOcrFrame ocr_frame(img_ori, db_detect_out);
        auto glab_out_opt = ocr_frame();

        if (!glab_out_opt) {
            return std::nullopt;
        }
        std::vector<deepvac::AggressiveBox> glab_result = glab_out_opt.value();
        for(int i=0; i<glab_result.size(); ++i){
            result.push_back(glab_result[i].getRect());
        }
    } else{
        result.assign(db_detect_out.begin(), db_detect_out.end());
    }

    for(int i=0; i<result.size(); ++i){
        cv::RotatedRect box = result[i];
        if(is_extend_){
            if(box.size.width >= box.size.height){
                box.size.width += 2*crop_gap_;
            } else{
                box.size.height += 2*crop_gap_;
            }
        }
        cv::Mat img_crop = cropRect(img_ori, box);
        crop_imgs.push_back(img_crop);

        std::vector<int> rect_;
        cv::Mat crop_box;
        cv::boxPoints(box, crop_box);
        for (int row=0; row<crop_box.rows; row++) {
            for (int col=0; col<crop_box.cols; col++) {
                rect_.push_back(int(crop_box.at<float>(row, col)));
            }
        }
        rects.push_back(rect_);
    }
    std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> crop_imgs_and_rects(crop_imgs, rects);
    return crop_imgs_and_rects;
}

}//namespace
