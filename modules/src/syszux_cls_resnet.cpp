/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/
#include "syszux_cls_resnet.h"
#include "gemfield.h"

namespace deepvac{
void SyszuxClsResnet::set(std::vector<int> input_size) {
    input_size_ = input_size;
}

std::optional<std::pair<int, float>> SyszuxClsResnet::process(cv::Mat frame){
    cv::Mat input_frame = frame.clone();
    cv::cvtColor(input_frame, input_frame, cv::COLOR_BGR2RGB);
    cv::resize(input_frame, input_frame, cv::Size(input_size_[0], input_size_[1]));

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(input_frame, gemfield_org::NORMALIZE0_1, gemfield_org::MEAN_STD_FROM_IMAGENET, device_);
    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    
    auto output = forward(input_tensor);
    
    auto softmaxs = output.softmax(1);
    std::tuple<torch::Tensor, torch::Tensor> max_res = torch::max(softmaxs, 1);
    
    auto max_probability = std::get<0>(max_res).item<float>();
    auto index = std::get<1>(max_res).item<int>();

    std::pair result(index, max_probability);
    return result;
}
} //namespace deepvac
