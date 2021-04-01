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
    setBinding(4);
}

SyszuxFaceRetinaNV::SyszuxFaceRetinaNV(std::vector<unsigned char>&& buffer, std::string device):DeepvacNV(std::move(buffer), device){
    initParameter(device);
    setBinding(4);
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

    cv::Mat dst;
    frame.convertTo(dst, CV_32F);

    auto prior_nums = calculatePriorBox(h, w);
    setDynamicInputOutput((float*)dst.data, c, h, w);

    std::vector<void*> predicitonBindings = {datas_[0].deviceBuffer.data(), datas_[1].deviceBuffer.data(), datas_[2].deviceBuffer.data(), datas_[3].deviceBuffer.data()};
    auto predict = forward(predicitonBindings.data());

    for(int i = 1; i < 4; ++i) {
        cudaMemcpy(datas_[i].hostBuffer.data(), datas_[i].deviceBuffer.data(), datas_[i].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
    }
    //Nx4    //Nx2    //Nx10
    auto loc = torch::from_blob(datas_[1].hostBuffer.data(), {1, prior_nums, 4}).to(device_);//output[0].toTensor();
    auto forward_conf = torch::from_blob(datas_[2].hostBuffer.data(), {1, prior_nums, 2}).to(device_);//output[1].toTensor();
    auto landms = torch::from_blob(datas_[3].hostBuffer.data(), {1, prior_nums, 10}).to(device_);//output[2].toTensor();

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
        faces_info.emplace_back(std::tuple(dst_img, bbox_vec, dst_points));}

    return faces_info;
}

void SyszuxFaceRetinaNV::setDynamicInputOutput(float* data, const int inputC, const int inputH, const int inputW) {
    //input
    datas_[0].hostBuffer.resize(nvinfer1::Dims4{1, inputC, inputH, inputW});
    datas_[0].deviceBuffer.resize(nvinfer1::Dims4{1, inputC, inputH, inputW});
    auto hostDataBuffer = static_cast<float*>(datas_[0].hostBuffer.data());
    int mean[] = {104, 117, 123};
    for (int c = 0; c < inputC; ++c) {
        for(int j = 0, volChl=inputH*inputW; j < volChl; ++j) {
            hostDataBuffer[c*volChl + j] = data[j*inputC + c] - mean[c];
        }
    }
    cudaMemcpy(datas_[0].deviceBuffer.data(), datas_[0].hostBuffer.data(), datas_[0].hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
    trt_context_->setBindingDimensions(0, nvinfer1::Dims4{1, inputC, inputH, inputW});
    //output
    auto nums = calculatePriorBox(inputH, inputW);
    int c[] = {0, 4, 2, 10};
    for(int i = 1; i < 4; ++i) {
        datas_[i].hostBuffer.resize(nvinfer1::Dims3{1, nums, c[i]});
        datas_[i].deviceBuffer.resize(nvinfer1::Dims3{1, nums, c[i]});
    }
}

int SyszuxFaceRetinaNV::calculatePriorBox(const int h, const int w) {
    std::vector<int> steps_{8,16,32};
    std::vector<std::vector<int>> feature_maps;

    for(int i=0; i<steps_.size(); i++){
        feature_maps.push_back({ static_cast<int>(std::ceil(1.0 * h/steps_[i])), static_cast<int>(std::ceil(1.0 * w/steps_[i])) });
    }

    int nums = 0;
    for(int i = 0; i < feature_maps.size(); ++i) {
        int fh = feature_maps[i][0];
        int fw = feature_maps[i][1];
        nums += fh * fw * 2;
    }
    return nums;    
}

} //namespace deepvac
