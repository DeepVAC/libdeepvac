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

std::optional<at::Tensor> cvMat2Tensor(cv::Mat&& tmp_frame, NORMALIZE_TYPE normalize, MEAN_STD_TYPE mean_std){
    cv::Mat frame = tmp_frame.clone();
    auto t_opt = cvMat2Tensor(frame, normalize, mean_std);
    if (!t_opt) {
        return std::nullopt;
    }
    auto t = t_opt.value();
    return t.clone();
} 

std::optional<at::Tensor> cvMat2Tensor(cv::Mat& frame, NORMALIZE_TYPE normalize, MEAN_STD_TYPE mean_std){
    if (frame.rows == 0 or frame.cols == 0){
        GEMFIELD_E("illegal img: wrong rows or cols.");
        return std::nullopt;
    }
    switch(normalize) {
        case NO_NORMALIZE :
            frame.convertTo(frame, CV_32FC3);
            break;
        case NORMALIZE0_1 :
            frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
            break;
        case NORMALIZE_1_1 :
            frame.convertTo(frame, CV_32FC3, 1.0f / 127.5f, -1.0f);
            break;
        default:
            GEMFIELD_E("invalid selection of normalize.");
            return std::nullopt;
    }
    auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    switch(mean_std){
        case NO_MEAN_STD :
            break;
        case MEAN_STD_FROM_IMAGENET :
            input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
            input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
            input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
            break;
        case MEAN_STD_FROM_FACE :
            input_tensor[0][0] = input_tensor[0][0].sub_(104);
            input_tensor[0][1] = input_tensor[0][1].sub_(117);
            input_tensor[0][2] = input_tensor[0][2].sub_(123);
            break;
        default:
            GEMFIELD_E("invalid selection of mean_std.");
            return std::nullopt;
    }
    return input_tensor;
}

std::optional<at::Tensor> cvMat2Tensor(std::vector<cv::Mat>& frames, NORMALIZE_TYPE normalize, MEAN_STD_TYPE mean_std){
    if (frames.size() == 0) {
        GEMFIELD_E("illegal vector: wrong size (size = 0).");
        return std::nullopt;
    }
    int h = frames[0].rows;
    int w = frames[0].cols;
    for (auto frame : frames) {
        if (frame.rows == 0 or frame.cols == 0){
            GEMFIELD_E("illegal img: wrong rows or cols.");
            return std::nullopt;
        }
        if (frame.rows != h or frame.cols != w){
            GEMFIELD_E("illegal img: rows or cols are not consistent.");
            return std::nullopt;
	}
    }
    cv::Mat batch_image;
    cv::vconcat(frames, batch_image);
    
    switch(normalize) {
        case NO_NORMALIZE :
            batch_image.convertTo(batch_image, CV_32FC3);
            break;
        case NORMALIZE0_1 :
            batch_image.convertTo(batch_image, CV_32FC3, 1.0f / 255.0f);
            break;
        case NORMALIZE_1_1 :
            batch_image.convertTo(batch_image, CV_32FC3, 1.0f / 127.5f, -1.0f);
            break;
        default:
            GEMFIELD_E("invalid selection of normalize.");
            return std::nullopt;
    }

    auto input_tensor = torch::from_blob(batch_image.data, {(int)frames.size(), h, w, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});

    switch(mean_std){
        case NO_MEAN_STD :
            break;
        case MEAN_STD_FROM_IMAGENET :
            input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
            input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
            input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
            break;
        case MEAN_STD_FROM_FACE :
            input_tensor[0][0] = input_tensor[0][0].sub_(104);
            input_tensor[0][1] = input_tensor[0][1].sub_(117);
            input_tensor[0][2] = input_tensor[0][2].sub_(123);
            break;
        default:
            GEMFIELD_E("invalid selection of mean_std.");
            return std::nullopt;
    }
    return input_tensor.clone();
}

std::optional<at::Tensor> img2Tensor(std::string& img_path, bool is_rgb, NORMALIZE_TYPE normalize, MEAN_STD_TYPE mean_std){
    auto frame_opt = img2CvMat(img_path, is_rgb);
    if(!frame_opt){
        return std::nullopt;
    }
    cv::Mat frame = frame_opt.value();
    return cvMat2Tensor(std::move(frame), normalize, mean_std);
}
}//namespace
