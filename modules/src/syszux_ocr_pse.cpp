/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "opencv2/opencv.hpp"
#include "syszux_ocr_pse.h"
#include "syszux_glab.h"

namespace deepvac {

void SyszuxOcrPse::set(int long_size, int crop_gap, int text_min_area, float text_mean_score) {
    long_size_ = long_size;
    crop_gap_ = crop_gap;
    text_min_area_ = text_min_area;
    text_mean_score_ = text_mean_score;
}

void SyszuxOcrPse::setGlab(bool is_glab){
    is_glab_ = is_glab;
}

cv::Mat SyszuxOcrPse::cropRect(cv::Mat &img, cv::RotatedRect &rotated_rects) {
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

std::optional< std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> > SyszuxOcrPse::process(cv::Mat img)
{
    GEMFIELD_SI;
    //prepare input
    std::vector<cv::Mat> crop_imgs;
    std::vector<std::vector<int>> rects;
    cv::Mat img_ori = img.clone();
    cv::Mat resize_img, rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    float scale1 = long_size_ * 1.0 / std::max(img.rows, img.cols);
    cv::resize(rgb_img, resize_img, cv::Size(), scale1, scale1);
    
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(resize_img, gemfield_org::NORMALIZE0_1, gemfield_org::MEAN_STD_FROM_IMAGENET);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();

    //prepare forward
    auto outputs = forward(input_tensor);

    //prepare output
    outputs = outputs.to(device_);
    outputs = outputs.squeeze();
    auto scores = torch::sigmoid(outputs.select(0, 0));
    outputs = torch::sign(outputs.sub_(1.0));
    outputs = outputs.add_(1).div_(2);
    auto text = outputs.select(0, 0);

    // kernel_num can be 3 or 7
    int kernel_num = 7;
    auto kernels = outputs.slice(0, 0, kernel_num) * text;
    kernels = kernels.toType(torch::kU8);
    
    float min_area = 10.0;
    
    auto pred = adaptorPse(kernels, min_area);
    std::vector<float> scale2 = {(float)(img.cols * 1.0 / pred[0].size()), (float)(img.rows * 1.0 / pred.size())};
    torch::Tensor label = torch::randn({(int)pred.size(), (int)pred[0].size()});
    for (int i=0; i<pred.size(); i++){
        label[i] = torch::tensor(pred[i]);
    }
    int label_num = torch::max(label).item<int>() + 1;

    std::vector<cv::RotatedRect> pse_detect_out;
    for(int i=1; i<label_num; i++)
    {
        torch::Tensor mask_index = (label==i);
        torch::Tensor points = torch::nonzero(mask_index);
        torch::Tensor temp = points.select(1, 0).clone();
        points.select(1, 0) = points.select(1, 1);
        points.select(1, 1) = temp;
        if (points.size(0) <= text_min_area_){
            continue;
        }

        torch::Tensor scores_i = scores.masked_select(mask_index);
        auto score_mean = torch::mean(scores_i).item<float>();
        if (score_mean < text_mean_score_){
            continue;
        }

        points = points.toType(torch::kFloat);
        points = points.to(torch::kCPU);
        cv::Mat points_mat(points.size(0), points.size(1), CV_32FC1);
        std::memcpy((void *) points_mat.data, points.data_ptr(), torch::elementSize(torch::kFloat) * points.numel());
        cv::RotatedRect rect = cv::minAreaRect(points_mat);

        cv::Point2f center = rect.center;
        cv::Size2f size = rect.size;
        float angle = rect.angle;
        rect.center.x = rect.center.x * scale2[0];
        rect.center.y = rect.center.y * scale2[1];
        rect.size.width = rect.size.width * scale2[0];
        rect.size.height = rect.size.height * scale2[1];
        pse_detect_out.push_back(rect);
    }

    std::vector<cv::RotatedRect> result;
    if(is_glab_){
        deepvac::DeepvacOcrFrame ocr_frame(img_ori, pse_detect_out);
        auto glab_out_opt = ocr_frame();

        if (!glab_out_opt) {
            return std::nullopt;
        }
        std::vector<deepvac::AggressiveBox> glab_result = glab_out_opt.value();
        for(int i=0; i<glab_result.size(); ++i){
            result.push_back(glab_result[i].getRect());
        }
    }
    else{
        result.assign(pse_detect_out.begin(), pse_detect_out.end());
    }

    for(int i=0; i<result.size(); ++i){
        cv::RotatedRect box = result[i];
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

std::vector<std::vector<float>> SyszuxOcrPse::merge(std::vector<std::vector<float>> rects) {
    std::vector<std::vector<float>> keep;
    for (int i=0; i<rects.size(); i++) {
        bool flag = false;
        for (int j=0; j<keep.size(); j++) {
            if (isMerge(rects[i], keep[j])) {
                flag = true;
                float x_min = std::min(rects[i][0], keep[j][0]);
                float y_min = std::min(rects[i][1], keep[j][1]);
                float x_max = std::max(rects[i][2], keep[j][2]);
                float y_max = std::max(rects[i][3], keep[j][3]);
                keep[j] = {x_min, y_min, x_max, y_max};
                break;
            }
        }
        if (flag == false) {
            keep.push_back(rects[i]);
        }
    }
    return keep;
}

std::vector<std::vector<float>> SyszuxOcrPse::mergeBox(std::vector<std::vector<float>> rects) {
    std::vector<std::vector<float>> keep = rects;
    int length = keep.size();
    while (true) {
        keep = merge(keep);
        if (keep.size() == length) {
            break;
        } else {
            length = keep.size();
        }
    }
    return keep;
}

bool SyszuxOcrPse::isMerge(std::vector<float> rect1, std::vector<float> rect2) {
    float x1_min = rect1[0];
    float y1_min = rect1[1];
    float x1_max = rect1[2];
    float y1_max = rect1[3];
    float x2_min = rect2[0];
    float y2_min = rect2[1];
    float x2_max = rect2[2];
    float y2_max = rect2[3];
    

    if (y1_max <= y2_min || y1_min >= y2_max) {
        return false;
    }

    float y[4] = {y1_min, y1_max, y2_min, y2_max};
    std::sort(y, y + 4);
    float x_thre = 2 * (y[3] - y[0]);
    if ((y[2]-y[1])/(y[3]-y[0]) < 0.7) {
        return false;
    }
    if ( (((x1_min - x2_max)>=0) && ((x1_min - x2_max)>x_thre)) || (((x2_min - x1_max)>=0) && ((x2_min - x1_max)>x_thre)) ) {
        return false;
    }
    return true;
}

void SyszuxOcrPse::getKernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals) {
    for (int i = 0; i < input_data.size(0); ++i) {
        cv::Mat kernal(input_data[i].size(0), input_data[i].size(1), CV_8UC1);
        std::memcpy((void *) kernal.data, input_data[i].data_ptr(), sizeof(torch::kU8) * input_data[i].numel());
        kernals.emplace_back(kernal);
    }
}

void SyszuxOcrPse::growingTextLine(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area) {
    cv::Mat label_mat;
    int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);

    int area[label_num + 1];
    memset(area, 0, sizeof(area));
    for (int x = 0; x < label_mat.rows; ++x) {
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            area[label] += 1;
        }
    }

    std::queue<cv::Point> queue, next_queue;
    for (int x = 0; x < label_mat.rows; ++x) {
        std::vector<int> row(label_mat.cols);
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);

            if (label == 0) continue;
            if (area[label] < min_area) continue;

            cv::Point point(x, y);
            queue.push(point);
            row[y] = label;
        }
        text_line.emplace_back(row);
    }

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
        while (!queue.empty()) {
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int label = text_line[x][y];

            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int tmp_x = x + dx[d];
                int tmp_y = y + dy[d];

                if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                if (text_line[tmp_x][tmp_y] > 0) continue;

                cv::Point point(tmp_x, tmp_y);
                queue.push(point);
                text_line[tmp_x][tmp_y] = label;
                is_edge = false;
            }

            if (is_edge) {
                next_queue.push(point);
            }
        }
        swap(queue, next_queue);
    }
}

std::vector<std::vector<int>> SyszuxOcrPse::adaptorPse(torch::Tensor input_data, float min_area) {
    std::vector<cv::Mat> kernals;
    input_data = input_data.to(torch::kCPU);
    getKernals(input_data, kernals);

    std::vector<std::vector<int>> text_line;
    growingTextLine(kernals, text_line, min_area);
    return text_line;
}
}//namespace
