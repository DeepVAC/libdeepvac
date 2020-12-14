/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_cls_mobile.h"

namespace deepvac{
SyszuxClsMobile::SyszuxClsMobile(std::string path, std::string device):Deepvac(path, device){
}

std::optional<std::pair<int, float>> SyszuxClsMobile::process(cv::Mat frame){
    cv::Mat input_img = frame.clone();
    input_img.convertTo(input_img, CV_32FC3, 1 / 127.5, -1);
    
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(input_img, false);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    input_tensor = input_tensor.to(device_);

    auto pred = forward(input_tensor);
    auto softmaxs = pred.softmax(1);
    std::tuple<torch::Tensor, torch::Tensor> max_res = torch::max(softmaxs, 1);
    auto max_probability = std::get<0>(max_res).item<float>();
    auto index = std::get<1>(max_res).item<int>();

    std::pair result(index, max_probability);
    return result;
}

std::optional<std::vector<std::pair<int, float>>> SyszuxClsMobile::process(std::vector<cv::Mat> frames){
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(frames, false);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    input_tensor = input_tensor.toType(torch::kFloat);
    input_tensor = input_tensor.to(device_);
    input_tensor = input_tensor.div_(127.5).sub_(1);

    auto pred = forward(input_tensor);
    auto softmaxs = pred.softmax(1);
    std::vector<std::pair<int, float>> results;
    for (int i=0; i<softmaxs.size(0); i++) {
        std::tuple<torch::Tensor, torch::Tensor> max_res = torch::max(softmaxs[i], 0);
        auto max_probability = std::get<0>(max_res).item<float>();
        auto index = std::get<1>(max_res).item<int>();
        std::pair result(index, max_probability);
	results.push_back(result);
    }
    return results;
}
}//namespace
