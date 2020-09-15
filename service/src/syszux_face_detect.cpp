/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include "syszux_face_detect.h"

namespace deepvac{

SyszuxFaceDetect::SyszuxFaceDetect(Deepvac&& deepvac): deepvac_(std::move(deepvac)), prior_box_({{16,32},{64,128},{256,512}}, {8,16,32}){
    device_ = deepvac_.getDevice();
}

std::optional<at::Tensor> SyszuxFaceDetect::operator()(cv::Mat frame){
    int h = frame.cols;
    int w = frame.rows;
    int c = frame.channels();
    int max_edge = std::max(h, w);
    int max_hw = 2000;

    if(max_edge > max_hw){
        cv::resize(frame, frame, cv::Size(int(w*max_hw/max_edge), int(h*max_hw/max_edge)));
    }

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(frame, true);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    //input_tensor -= (104, 117, 123); change this accordingly
    auto output = deepvac_.forwardTuple(input_tensor);
    //Nx4    //Nx2    //Nx10
    auto loc = output[0].toTensor();
    auto forward_conf = output[1].toTensor();
    auto landms = output[2].toTensor();

    std::cout<<loc.sizes()<<"\t :"<<loc<<std::endl;
    std::cout<<forward_conf.sizes()<<"\t :"<<forward_conf<<std::endl;
    std::cout<<landms.sizes()<<"\t :"<<landms<<std::endl;

    //gemfield
    torch::Tensor prior_output = prior_box_.forward({frame.rows, frame.cols});
    prior_output = prior_output.to(device_);
    return prior_output;

    loc = loc.to(device_);
    forward_conf = forward_conf.to(device_);
    landms = landms.to(device_);

    torch::Tensor boxes;

    int resize = 1;

    torch::Tensor variances_tensor = torch::tensor({0.1, 0.2});
    variances_tensor = variances_tensor.to(device_);
    //gemfield
}

} //namespace deepvac