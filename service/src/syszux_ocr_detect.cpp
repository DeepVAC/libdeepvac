/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "opencv2/opencv.hpp"
#include "syszux_ocr_detect.h"

namespace deepvac {

SyszuxOcrDetect::SyszuxOcrDetect(std::string device):Deepvac("/gemfield/hostpv/gemfield/pse/pse1.deepvac", device) {}

void SyszuxOcrDetect::set(int long_size, int crop_gap) {
    long_size_ = long_size;
    crop_gap_ = crop_gap;
}

std::optional<std::vector<cv::Mat>> SyszuxOcrDetect::operator() (cv::Mat img)
{
    std::vector<cv::Mat> crop_imgs;
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

    auto outputs = forward(tensor_img);
    outputs = outputs.squeeze();
    auto scores = torch::sigmoid(outputs.select(0, 0));
    outputs = torch::sign(outputs.sub_(1.0));
    outputs = outputs.add_(1).div_(2);

    auto text = outputs.select(0, 0);
    auto kernels = outputs.slice(0, 0, 3) * text;
    kernels = kernels.toType(torch::kU8);
    
    float min_area = 10.0;
    auto pred = adaptor_pse(kernels, min_area);
    std::vector<float> scale2 = {(float)(img.cols * 1.0 / pred[0].size()), (float)(img.rows * 1.0 / pred.size())};
    torch::Tensor label = torch::randn({(int)pred.size(), (int)pred[0].size()});
    for (int i=0; i<pred.size(); i++){
        label[i] = torch::tensor(pred[i]);
    }
    int label_num = torch::max(label).item<int>() + 1;
    for(int i=1; i<label_num; i++){
        torch::Tensor mask_index = (label==i);
        torch::Tensor points = torch::nonzero(mask_index);
        torch::Tensor temp = points.select(1, 0).clone();
        points.select(1, 0) = points.select(1, 1);
        points.select(1, 1) = temp;
        if (points.size(0) <= 300){
            continue;
        }

        torch::Tensor scores_i = scores.masked_select(mask_index);
        auto score_mean = torch::mean(scores_i).item<float>();
        if (score_mean < 0.93){
            continue;
        }

        points = points.toType(torch::kFloat);
        cv::Mat points_mat(points.size(0), points.size(1), CV_32FC1);
        std::memcpy((void *) points_mat.data, points.data_ptr(), torch::elementSize(torch::kFloat) * points.numel());
        auto rect = cv::minAreaRect(points_mat);
        cv::Mat crop_box;
        cv::boxPoints(rect, crop_box);
        
        auto crop_box_tensor = torch::from_blob(crop_box.data, {crop_box.rows, crop_box.cols}).toType(torch::kFloat);
        crop_box_tensor.select(1, 0) = crop_box_tensor.select(1, 0).mul_(scale2[0]);
        crop_box_tensor.select(1, 1) = crop_box_tensor.select(1, 1).mul_(scale2[1]);
        crop_box_tensor.select(1, 0) = crop_box_tensor.select(1, 0).clamp_(0, img.cols);
        crop_box_tensor.select(1, 1) = crop_box_tensor.select(1, 1).clamp_(0, img.rows);
        	
        auto max_tensor = std::get<0>(torch::max(crop_box_tensor, 0));
        auto min_tensor = std::get<0>(torch::min(crop_box_tensor, 0));
        int x_max = max_tensor[0].item().toInt();
        int y_max = max_tensor[1].item().toInt();
        int x_min = min_tensor[0].item().toInt();
        int y_min = min_tensor[1].item().toInt();
       
      	x_max = (x_max + crop_gap_) >= img.cols ? img.cols : (x_max + crop_gap_);
        x_min = (x_min - crop_gap_) <= 0 ? 0 : (x_min - crop_gap_);
        
        auto crop_img = img(cv::Rect(x_min, y_min, x_max-x_min, y_max-y_min));
        crop_imgs.push_back(crop_img);
    }
    return crop_imgs;
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
    input_data = input_data.to(torch::kCPU);
    get_kernals(input_data, kernals);

    std::vector<std::vector<int>> text_line;
    growing_text_line(kernals, text_line, min_area);
    return text_line;
}
}//namespace
