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
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    }
    return frame;
}

std::optional<at::Tensor> cvMat2Tensor(cv::Mat& frame, bool is_normalize){
    if (frame.rows == 0 or frame.cols == 0){
        GEMFIELD_E("illegal img: wrong rows or cols.");
        return std::nullopt;
    }
    if(is_normalize){
        frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    }
    auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    return input_tensor;
}

std::optional<at::Tensor> cvMat2Tensor(std::vector<cv::Mat>& frames, bool is_normalize){
    if (frames.size() == 0) {
        GEMFIELD_E("illegal vector: wrong size (size = 0).");
        return std::nullopt;
    }
    for (auto frame : frames) {
        if (frame.rows == 0 or frame.cols == 0){
            GEMFIELD_E("illegal img: wrong rows or cols.");
            return std::nullopt;
	}
    }
    cv::Mat batch_image;
    cv::vconcat(frames, batch_image);
    if(is_normalize){
        batch_image.convertTo(batch_image, CV_32FC3, 1.0f / 255.0f);
    }
    else {
        batch_image.convertTo(batch_image, CV_32FC3);
    }
    auto input_tensor = torch::from_blob(batch_image.data, {(int)frames.size(), frames[0].rows, frames[0].cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    return input_tensor.clone();
}

std::optional<at::Tensor> img2Tensor(std::string& img_path, bool is_rgb, bool is_normalize){
    auto frame_opt = img2CvMat(img_path, is_rgb);
    if(!frame_opt){
        return std::nullopt;
    }
    cv::Mat frame = frame_opt.value();
    return cvMat2Tensor(frame,is_normalize);
}
}//namespace
