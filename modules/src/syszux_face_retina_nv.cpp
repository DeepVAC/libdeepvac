/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/
#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "NvInfer.h"
#include "syszux_face_retina_nv.h"
#include "gemfield.h"

namespace deepvac{
SyszuxFaceRetinaNV::SyszuxFaceRetinaNV(std::string path, std::string device):DeepvacNV(path, device) {
    initParameter(device);
    setBinding(trt_module_->getNbBindings());
}

SyszuxFaceRetinaNV::SyszuxFaceRetinaNV(std::vector<unsigned char>&& buffer, std::string device):DeepvacNV(std::move(buffer), device){
    initParameter(device);
    setBinding(trt_module_->getNbBindings());
}

std::optional<std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>>> SyszuxFaceRetinaNV::process(cv::Mat frame){
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

    auto input_tensor_opt = gemfield_org::cvMat2Tensor(std::move(frame), gemfield_org::NO_NORMALIZE, gemfield_org::MEAN_STD_FROM_FACE, device_);
    if(!input_tensor_opt){
        return std::nullopt;
    }

    auto predicitonBindings = prepareInputOutput(input_tensor_opt.value());
    auto predict = forward(predicitonBindings.data());
    torch::Tensor loc, landms, forward_conf;
    for(int i = 1; i < 4; ++i) {
        auto channel = datas_[i].deviceBuffer.shape()[2];
        if(4 == channel) {
            loc = datas_[i].deviceBuffer.toTensor();
        } else if(2 == channel) {
            forward_conf = datas_[i].deviceBuffer.toTensor();
        } else if(10 == channel) {
            landms = datas_[i].deviceBuffer.toTensor();
        } else {
            GEMFIELD_E("face detect model error, invalid output dims");
        }
    }

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
    if (index[0].size(0) == 0) {
        return std::nullopt;
    }

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
    if(dets.size(0) == 0){
        return std::nullopt;
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

    std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>> faces_info;
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
