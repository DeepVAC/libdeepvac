/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "opencv2/opencv.hpp"
#include "syszux_ocr_detect.h"

namespace deepvac {

SyszuxOcrDetect::SyszuxOcrDetect(int long_size, std::string device):Deepvac("/gemfield/hostpv/gemfield/pse/pse1.deepvac", device),
    long_size_(long_size){}

std::optional<cv::Mat> SyszuxOcrDetect::operator() (cv::Mat img)
{
    cv::Mat resize_img, rgb_img;
    cv::Mat text_box = img.clone();
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    float scale1 = long_size_ * 1.0 / std::max(img.rows, img.cols);
    cv::resize(rgb_img, resize_img, cv::Size(), scale1, scale1);
    
    auto tensor_img = torch::from_blob(resize_img.data, {1, resize_img.rows, resize_img.cols, resize_img.channels()}, torch::kByte);
    tensor_img = tensor_img.to(device_);
    tensor_img = tensor_img.permute({0, 3, 1, 2});
    tensor_img = tensor_img.toType(torch::kFloat);
    tensor_img = tensor_img.div(255);
    tensor_img[0][0] = tensor_img[0][0].sub_(0.485).div_(0.229);
    tensor_img[0][1] = tensor_img[0][1].sub_(0.456).div_(0.224);
    tensor_img[0][2] = tensor_img[0][2].sub_(0.406).div_(0.225);

    torch::Tensor kernels = torch::ones({3, 120, 640});
    kernels = kernels.toType(torch::kU8);
    torch::Tensor scores = torch::randn({120, 640});
    torch::Tensor text = torch::randn({120, 640});
    float min_area = 10.0;
    auto pred = adaptor_pse(kernels, min_area);
    std::vector<float> scale2 = {(float)(img.cols * 1.0 / pred[0].size()), (float)(img.rows * 1.0 / pred.size())};
    torch::Tensor label = torch::randn({(int)pred.size(), (int)pred[0].size()});
    for (int i=0; i<pred.size(); i++){
        label[i] = torch::tensor(pred[i]);
    }
    std::vector<std::vector<std::vector<cv::Point>>> bboxes;
    int label_num = torch::max(label).item<int>() + 1;
    for(int i=0; i<label_num; i++)
    {
        torch::Tensor mask_index = (label==i);
        torch::Tensor scores_i = scores.masked_select(mask_index);
        if (scores_i.size(0) <= 300){
            continue;
        }
        auto score_mean = torch::mean(scores_i).item<float>();
        if (score_mean < 0.93){
            continue;
        }
        torch::Tensor binary = torch::zeros(label.sizes()).toType(torch::kU8);
        torch::Tensor a = torch::ones(label.sizes()).toType(torch::kU8);
        binary = torch::where(label==i, a, binary);
        cv::Mat binary_mat(binary.size(0), binary.size(1), CV_8UC1);
        std::memcpy((void *) binary_mat.data, binary.data_ptr(), torch::elementSize(torch::kU8) * binary.numel());

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        auto contour = contours[0];
        if (contour.size() <= 2){
            continue;
        }
        //std::vector<int> bbox;
        //for (int i=0; i<contour.size(); i++){
        //    bbox.push_back(contour[i].x * scale2[0]);
        //    bbox.push_back(contour[i].y * scale2[1]);
        //}
        bboxes.push_back(contours);
    }
    for(int i=0; i<bboxes.size(); i++){
        cv::drawContours(text_box, bboxes[i], -1, (0, 255, 0), 2);
    }
    cv::resize(text_box, text_box, cv::Size(text.size(1), text.size(0)));
    return text_box;
}

void SyszuxOcrDetect::get_kernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals) {
    for (int i = 0; i < input_data.size(0); ++i) {
        cv::Mat kernal(input_data[i].size(0), input_data[i].size(1), CV_8UC1);
        std::memcpy((void *) kernal.data, input_data[i].data_ptr(), sizeof(torch::kU8) * input_data[i].numel());
        kernals.emplace_back(kernal);
    }
}

void SyszuxOcrDetect::growing_text_line(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area) {
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

std::vector<std::vector<int>> SyszuxOcrDetect::adaptor_pse(torch::Tensor input_data, float min_area) {
    std::vector<cv::Mat> kernals;
    get_kernals(input_data, kernals);

    std::vector<std::vector<int>> text_line;
    growing_text_line(kernals, text_line, min_area);
    return text_line;
}
}//namespace
