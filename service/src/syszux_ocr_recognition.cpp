/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "syszux_ocr_recognition.h"

SyszuxOcrRecognition::SyszuxOcrRecognition(std::string device){
    device_ = device;
}

void SyszuxOcrRecognition::operator () (cv::Mat img){
    int h = img.rows;
    int w = img.cols;
    
    int in_w = (int)((float)32 / (float)h * (float)w);
    in_w = (int)(in_w / 2);
    in_w -= in_w % 4;
    cv::Mat resize_img, border_img;
    cv::resize(img, resize_img, cv::Size(in_w, 32));
    int img_width = 2 * in_w;
    int left = (int)((img_width - in_w) / 2);
    int right = img_width - in_w - left;

    resize_img.convertTo(resize_img, CV_32F, 1 / 127.5, -1);
    cv::copyMakeBorder(resize_img, border_img, 0, 0, left, right, cv::BORDER_CONSTANT, {0, 0, 0});

    auto tensor_img = torch::from_blob(border_img.data, {border_img.rows, border_img.cols, border_img.channels()});
    tensor_img = tensor_img.to(device_);
    tensor_img = tensor_img.toType(torch::kFloat);
    tensor_img = tensor_img.permute({2, 0, 1});
    std::cout << tensor_img << std::endl;
}
