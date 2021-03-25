/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "syszux_face_retina.h"
#include "gemfield.h"

namespace deepvac{

SyszuxFaceRetina::SyszuxFaceRetina(std::string path, std::string device):Deepvac(path, device), prior_box_({{16,32},{64,128},{256,512}}, {8,16,32}){
    initParameter(device);
}

SyszuxFaceRetina::SyszuxFaceRetina(std::vector<unsigned char>&& buffer, std::string device):Deepvac(std::move(buffer), device), prior_box_({{16,32},{64,128},{256,512}}, {8,16,32}){
    initParameter(device);
}

void SyszuxFaceRetina::initParameter(std::string device){
    variances_tensor_ = torch::tensor({0.1, 0.2}).to(device);

    setTopK(50);
    setKeepTopK(50);
    setConfThreshold(0.4);
    setNMSThreshold(0.4);
    setMaxHW(2000);
    setGapThreshold(0.1);

    last_w_ = 0;
    last_h_ = 0;
    last_prior_ = torch::ones({1, 4});
    last_box_scale_ = torch::ones({1, 4});
    last_lmk_scale_ = torch::ones({1, 10});
}

void SyszuxFaceRetina::setTopK(int top_k){
    top_k_ = top_k;
}

void SyszuxFaceRetina::setKeepTopK(int keep_top_k){
    keep_top_k_ = keep_top_k;
}

void SyszuxFaceRetina::setConfThreshold(float confidence_threshold){
    confidence_threshold_ = confidence_threshold;
}

void SyszuxFaceRetina::setNMSThreshold(float nms_threshold){
    nms_threshold_ = nms_threshold;
}

void SyszuxFaceRetina::setMaxHW(int max_hw){
    max_hw_ = max_hw;
}

void SyszuxFaceRetina::setGapThreshold(float gap_threshold){
    gap_threshold_ = gap_threshold;
}

std::optional<std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>>> SyszuxFaceRetina::process(cv::Mat frame){
    GEMFIELD_SI;
    //prepare input
    int h = frame.rows;
    int w = frame.cols;
    int c = frame.channels();
    int max_edge = std::max(h, w);
    if(max_edge > max_hw_){
        cv::resize(frame, frame, cv::Size(int(w*max_hw_/max_edge), int(h*max_hw_/max_edge)));
        h = frame.rows;
        w = frame.cols;
    }
    //gemfield, prepare output
    if ( std::abs(h-last_h_)<=gap_threshold_*last_h_ and std::abs(w-last_w_)<=gap_threshold_*last_w_) {
        if ( h!=last_h_ or w!=last_w_ ) {
            cv::resize(frame, frame, cv::Size(last_w_, last_h_));
        }
    } else {
        last_w_ = w;
        last_h_ = h;

        last_prior_ = prior_box_.forward({h, w});
        last_prior_ = last_prior_.to(device_);

        last_box_scale_ = torch::tensor({w, h, w, h});
        last_box_scale_ = last_box_scale_.to(device_);

        last_lmk_scale_ = torch::tensor({w, h, w, h, w, h, w, h, w, h});
        last_lmk_scale_ = last_lmk_scale_.to(device_);
    }

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(std::move(frame), gemfield_org::NO_NORMALIZE, gemfield_org::MEAN_STD_FROM_FACE);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    //forward
    auto output = forward<std::vector<c10::IValue>>(input_tensor);
    //Nx4    //Nx2    //Nx10
    auto loc = output[0].toTensor();
    auto forward_conf = output[1].toTensor();
    auto landms = output[2].toTensor();

    loc = loc.squeeze(0);
    forward_conf = forward_conf.squeeze(0);
    landms = landms.squeeze(0);

    float resize = 1.;

    //gemfield
    torch::Tensor boxes = gemfield_org::getDecodeBox(last_prior_, variances_tensor_, loc);
    boxes = torch::div(torch::mul(boxes, last_box_scale_), resize);

    gemfield_org::decodeLandmark(last_prior_, variances_tensor_, landms);
    landms = torch::div(torch::mul(landms, last_lmk_scale_), resize);

    torch::Tensor scores = forward_conf.slice(1, 1, 2);
    std::vector<torch::Tensor> index = torch::where(scores>confidence_threshold_);
    boxes = boxes.index({index[0]});
    landms = landms.index({index[0]});
    scores = scores.index({index[0]});

    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(scores, 0, 1);
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1);
    idx = idx.slice(0, 0, top_k_);

    boxes = boxes.index({idx});
    landms = landms.index({idx});
    scores = scores.index({idx});

    torch::Tensor dets = torch::cat({boxes, scores}, 1);
    torch::Tensor keep;
    keep = gemfield_org::nms(dets, nms_threshold_);

    // keep top-K faster NMS
    keep = keep.slice(0, 0, keep_top_k_);
    dets = dets.index({keep});
    landms = landms.index({keep});
    
    std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>> faces_info;

    if(dets.size(0) == 0){
        return faces_info;
    }

    std::string msg = gemfield_org::format("detected %d faces", dets.size(0));
    GEMFIELD_I(msg.c_str());

    if(dets.size(0) != landms.size(0)){
        std::string msg = gemfield_org::format("dets len mismatched landms len: %d vs %d", dets.size(0), landms.size(0));
        GEMFIELD_E(msg.c_str());
        return std::nullopt;
    }

    landms = landms.to(torch::kCPU);
    cv::Mat landms_mat(landms.size(0), landms.size(1), CV_32F);
    std::memcpy((void *) landms_mat.data, landms.data_ptr(), torch::elementSize(torch::kF32) * landms.numel());
    dets = dets.to(torch::kCPU);
    cv::Mat dets_mat(dets.size(0), dets.size(1), CV_32F);
    std::memcpy((void *) dets_mat.data, dets.data_ptr(), torch::elementSize(torch::kF32) * dets.numel());
    
    for(int i=0; i<landms_mat.rows; i++) {
        auto landmark = landms_mat.row(i);
        auto [dst_img, dst_points] = align_face_(frame, landmark);
        auto bbox = dets_mat.row(i);
        std::vector<float> bbox_vec(bbox.begin<float>(), bbox.end<float>());
        dst_img.convertTo(dst_img, CV_32FC3);
        faces_info.emplace_back(std::tuple(dst_img, bbox_vec, dst_points));
    }

    return faces_info;
}
} //namespace deepvac
