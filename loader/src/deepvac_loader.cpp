/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include <algorithm>
#include <cctype>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "deepvac_loader.h"
#include "gemfield.h"

namespace deepvac{

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
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    return frame;
}

std::optional<at::Tensor> img2tensor(std::string& img_path, bool is_rgb){
    cv::Mat frame = img2CvMat(img_path, is_rgb).value();
    auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor = input_tensor.sub_(0.5).div_(0.5);
    return input_tensor;
}

const std::string RecursiveFileIter::operator*() const {
    std::string p = (*walker_).path(); 
    return p;
}

const std::optional<cv::Mat> CvMatIter::operator*() const {
    std::string p = (*walker_).path();
    return img2CvMat(p);
}

const std::tuple<std::string, std::optional<at::Tensor>> ImgFileInputTensorPairIter::operator*() const {
    std::string p = (*walker_).path();
    std::string suffix = std::filesystem::path(p).extension();

    if (suffix.empty()){
        return {p, std::nullopt};
    }

    std::transform(suffix.begin(), suffix.end(), suffix.begin(),[](unsigned char c){ return std::tolower(c); });

    if(suffix_.find(suffix) == suffix_.end() ){
        return {p, std::nullopt};
    }

    return {p, img2tensor(p)};
}

//loader that based on std container.
std::optional<cv::Mat> DeepvacVectorIter::operator*() const {
    return img2CvMat(*walker_);
}

} //namespace deepvac