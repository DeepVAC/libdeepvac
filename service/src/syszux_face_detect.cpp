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
#include "gemfield.h"

namespace deepvac{

SyszuxFaceDetect::SyszuxFaceDetect(std::string device):Deepvac(facedet_deepvac, device),
    prior_box_({{16,32},{64,128},{256,512}}, {8,16,32}){}

std::optional<std::vector<cv::Mat>> SyszuxFaceDetect::operator()(cv::Mat frame){
    GEMFIELD_SI;
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
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(frame, false);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();
    input_tensor[0][0] = input_tensor[0][0].sub_(104);
    input_tensor[0][1] = input_tensor[0][1].sub_(117);
    input_tensor[0][2] = input_tensor[0][2].sub_(123);
    
    auto output = forwardTuple(input_tensor);
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

    float resize = 1.;

    torch::Tensor variances_tensor = torch::tensor({0.1, 0.2});
    variances_tensor = variances_tensor.to(device_);
    //gemfield
    torch::Tensor scale = torch::tensor({w, h, w, h});
    scale = scale.to(device_);
    torch::Tensor boxes = gemfield_org::getDecodeBox(prior_output, variances_tensor,loc);
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
    
    std::vector<cv::Mat> detect_vec;

    if(dets.size(0) == 0){
        return detect_vec;
    }

    std::string msg = gemfield_org::format("detected %d faces", dets.size(0));
    GEMFIELD_I(msg.c_str());

    if(dets.size(0) != landms.size(0)){
        std::string msg = gemfield_org::format("dets len mismatched landms len: %d vs %d", dets.size(0), landms.size(0));
        GEMFIELD_E(msg.c_str());
        return std::nullopt;
    }

    cv::Mat landms_mat(landms.size(0), landms.size(1), CV_32F);
    std::memcpy((void *) landms_mat.data, landms.data_ptr(), torch::elementSize(torch::kF32) * landms.numel());
    
    for(int i=0; i<landms_mat.rows; i++){
        auto landmark = landms_mat.row(i);
        cv::Mat dst_img = align_face_(frame_ori, landmark);
        detect_vec.push_back(dst_img);
    }
    return detect_vec;
}
} //namespace deepvac
