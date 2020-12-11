/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include <iostream>
#include <map>
#include "syszux_cls_mobile.h"

namespace deepvac{
SyszuxClsMobile::SyszuxClsMobile(std::string path, std::string device):Deepvac(path, device){
}

void SyszuxClsMobile::set(float confidence, float threshold) {
    confidence_ = confidence;
    threshold_ = threshold;
}

float SyszuxClsMobile::getScore(std::vector<float> confidences) {
    float numerator = 0.0;
    float denominator = 0.0;
    for (auto confidence : confidences) {
        if (confidence <= 0) {
            numerator = numerator + confidence * confidence;
        }
        denominator = denominator + confidence * confidence;
    }
    return numerator / denominator;
}

std::vector<cv::Mat> SyszuxClsMobile::getRotatedImgs(std::vector<cv::Mat> crop_imgs, std::vector<float> confidences) {
    float score = getScore(confidences);
    std::vector<cv::Mat> rotated_imgs;
    for (int i=0; i<crop_imgs.size(); i++) {
        if (std::abs(confidences[i])<0.99 && score>=threshold_ || confidences[i]<=-0.99) {
            cv::Mat rotated_img = crop_imgs[i].clone();
            cv::flip(rotated_img, rotated_img, -1);
            rotated_imgs.push_back(rotated_img);
        }
        else {
            rotated_imgs.push_back(crop_imgs[i].clone());
        }
    }
    return rotated_imgs;
}

std::optional<std::vector<cv::Mat>> SyszuxClsMobile::process(std::vector<cv::Mat> crop_imgs){
    if (crop_imgs.size()==1) {
        return crop_imgs;
    }
    std::vector<float> confidences;
    for (auto img : crop_imgs){
        int h = img.rows;
        int w = img.cols;
        std::vector<int> img_size = {192, 48, 3};

        int in_w = (int)(w/h * img_size[1]);
        cv::Mat resize_img;
        cv::resize(img, resize_img, cv::Size(in_w, img_size[1]));
	
        cv::Mat input_img;
        if (in_w >= img_size[0]) {
            cv::Rect rect(0, 0, img_size[0], img_size[1]);
            input_img = resize_img(rect);
        }
        else {
            cv::copyMakeBorder(resize_img, input_img, 0, 0, 0, img_size[0]-in_w, cv::BORDER_CONSTANT, {0, 0, 0});
        }
        input_img.convertTo(input_img, CV_32F, 1 / 127.5, -1);
        auto tensor_img = torch::from_blob(input_img.data, {1, input_img.rows, input_img.cols, input_img.channels()});
        tensor_img = tensor_img.to(device_);
        tensor_img = tensor_img.toType(torch::kFloat);
        tensor_img = tensor_img.permute({0, 3, 1, 2});
        
        auto pred = forward(tensor_img);
        auto softmaxs = pred.softmax(1);
        std::tuple<torch::Tensor, torch::Tensor> max_res = torch::max(softmaxs, 1);
        auto max_probability = std::get<0>(max_res).item<float>();
        auto index = std::get<1>(max_res).item<int>();
        max_probability = ((-2) * index + 1) * max_probability;

        confidences.push_back(max_probability);
    }
    auto rotated_imgs = getRotatedImgs(crop_imgs, confidences);

    return rotated_imgs;
}
}//namespace
