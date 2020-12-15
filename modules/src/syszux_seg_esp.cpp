/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "syszux_seg_esp.h"

namespace deepvac{

SyszuxSegEsp::SyszuxSegEsp(std::string path, std::string device):Deepvac(path, device){}

void SyszuxSegEsp::set(std::vector<int> image_size) {
    image_size_ = image_size;
}

std::optional<cv::Mat> SyszuxSegEsp::process(cv::Mat frame){
    int h = frame.rows;
    int w = frame.cols;
    int c = frame.channels();
    cv::Mat resize_img;
    cv::cvtColor(frame, resize_img, cv::COLOR_BGR2RGB);
    cv::resize(resize_img, resize_img, cv::Size(image_size_[0], image_size_[1]), cv::INTER_LINEAR);

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(resize_img, gemfield_org::NORMALIZE0_1, gemfield_org::MEAN_STD_FROM_IMAGENET);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    //forward
    auto output = forward(input_tensor);
    
    output = output.squeeze();
    auto out = output.argmax(0).cpu().to(torch::kByte);

    cv::Mat mask(out.size(0), out.size(1), CV_8UC1);
    std::memcpy((void *) mask.data, out.data_ptr(), torch::elementSize(torch::kU8) * out.numel());

    cv::resize(mask, mask, cv::Size(w, h), cv::INTER_LINEAR);
    
    return mask;
}
} //namespace deepvac
