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

std::optional<std::vector<cv::Mat>> SyszuxFaceDetect::operator()(cv::Mat frame){
    int h = frame.rows;
    int w = frame.cols;
    int c = frame.channels();
    int max_edge = std::max(h, w);
    int max_hw = 2000;
    if(max_edge > max_hw){
        cv::resize(frame, frame, cv::Size(int(w*max_hw/max_edge), int(h*max_hw/max_edge)));
        h = frame.rows;
        w = frame.cols;
    }
    cv::Mat frame_ori = frame.clone();

    frame.convertTo(frame, CV_32F);
    std::vector<cv::Mat> channels, src;
    cv::split(frame, channels);
    cv::Mat B = channels.at(0) - 104;
    cv::Mat G = channels.at(1) - 117;
    cv::Mat R = channels.at(2) - 123;
    src.push_back(B);
    src.push_back(G);
    src.push_back(R);
    cv::merge(src, frame);

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(frame, false);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    auto output = deepvac_.forwardTuple(input_tensor);
    //Nx4    //Nx2    //Nx10
    auto loc = output[0].toTensor();
    auto forward_conf = output[1].toTensor();
    auto landms = output[2].toTensor();
    //gemfield
    torch::Tensor prior_output = prior_box_.forward({frame.rows, frame.cols});
    prior_output = prior_output.to(device_);

    loc = loc.squeeze(0).to(device_);
    forward_conf = forward_conf.squeeze(0).to(device_);
    landms = landms.squeeze(0).to(device_);

    torch::Tensor boxes;
    float resize = 1.;

    torch::Tensor variances_tensor = torch::tensor({0.1, 0.2});
    variances_tensor = variances_tensor.to(device_);
    //gemfield
    torch::Tensor scale = torch::tensor({w, h, w, h});
    scale = scale.to(device_);
    gemfield_org::decodeBox(loc, prior_output, variances_tensor, boxes);
    boxes = torch::div(torch::mul(boxes, scale), resize);

    gemfield_org::decodeLandmark(prior_output, variances_tensor, landms);
    torch::Tensor scale1 = torch::tensor({w, h, w, h, w, h, w, h, w, h});
    scale1 = scale1.to(device_);
    landms = torch::div(torch::mul(landms, scale1), resize);

    torch::Tensor scores = forward_conf.slice(1, 1, 2);
    float confidence_threshold = 0.4;
    std::vector<torch::Tensor> index = torch::where(scores>confidence_threshold);
    boxes = boxes.index(index[0]);
    landms = landms.index(index[0]);
    scores = scores.index(index[0]);

    int top_k = 50;
    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(scores, 0, 1);
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1);
    idx = idx.slice(0, 0, top_k);

    boxes = boxes.index(idx);
    landms = landms.index(idx);
    scores = scores.index(idx);

    torch::Tensor dets = torch::cat({boxes, scores}, 1);
    float nms_threshold = 0.4;
    torch::Tensor keep;
    keep = gemfield_org::nms(dets, nms_threshold);

    // keep top-K faster NMS
    int keep_top_k = 50;
    keep = keep.slice(0, 0, keep_top_k);
    dets = dets.index(keep);
    landms = landms.index(keep);
    std::cout << "detected " << dets.size(0) << "faces@" << std::endl;

    if(dets.size(0)==0){
        return std::nullopt;
    }
    if(dets.size(0) != landms.size(0)){
        std::cout << "dets len mismatched landms len: " << dets.size(0) << " vs " << landms.size(0);
        return std::nullopt;
    }

    cv::Mat dets_mat(dets.size(0), dets.size(1), CV_32F);
    cv::Mat landms_mat(landms.size(0), landms.size(1), CV_32F);
    std::memcpy((void *) dets_mat.data, dets.data_ptr(), sizeof(torch::kF32) * dets.numel());
    std::memcpy((void *) landms_mat.data, landms.data_ptr(), 4*sizeof(torch::kF32) * landms.numel());
    
    std::vector<cv::Mat> detecte_out;
    for(int i=0; i<dets_mat.rows; i++){
        auto landmark = landms_mat.row(i);
	std::cout << landmark << std::endl;
        cv::Mat dst_img = align_face_(frame_ori, landmark);
        detecte_out.push_back(dst_img);
    }
    return detecte_out;
}
} //namespace deepvac
