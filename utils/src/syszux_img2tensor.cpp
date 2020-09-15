/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_img2tensor.h"
#include "gemfield.h"

namespace gemfield_org{
std::optional<cv::Mat> img2CvMat(std::string& img_path, bool is_rgb){
    cv::Mat frame;
    try{
        frame = cv::imread(img_path);
    }catch(std::exception& e){
        GEMFIELD_E2("Error to read img: ", img_path.c_str());
        return std::nullopt;
    }
    if (frame.rows == 0 or frame.cols == 0){
        GEMFIELD_E2("illegal img: ", img_path.c_str());
        return std::nullopt;
    }
    if(is_rgb){
        cv::cvtColor(frame, frame, CV_BGR2RGB);
    }
    return frame;
}

std::optional<at::Tensor> cvMat2Tensor(cv::Mat& frame){
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor = input_tensor.sub_(0.5).div_(0.5);
    return input_tensor;
}

std::optional<at::Tensor> img2Tensor(std::string& img_path, bool is_rgb){
    cv::Mat frame = img2CvMat(img_path, is_rgb).value();
    return cvMat2Tensor(frame);
}

}//namespace