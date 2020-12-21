/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_cls_mobile.h"

namespace deepvac{
std::optional<std::pair<int, float>> SyszuxClsMobile::process(cv::Mat frame){
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(std::move(frame), gemfield_org::NORMALIZE_1_1, gemfield_org::NO_MEAN_STD);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();

    auto pred = forward(input_tensor);
    auto softmaxs = pred.softmax(1);
    std::tuple<torch::Tensor, torch::Tensor> max_res = torch::max(softmaxs, 1);
    auto max_probability = std::get<0>(max_res).item<float>();
    auto index = std::get<1>(max_res).item<int>();

    std::pair result(index, max_probability);
    return result;
}

std::optional<std::vector<std::pair<int, float>>> SyszuxClsMobile::process(std::vector<cv::Mat> frames){
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(frames, gemfield_org::NORMALIZE_1_1, gemfield_org::NO_MEAN_STD);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();

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
