/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_detect_yolo.h"
#include <time.h>
#include <sys/time.h>
namespace deepvac{
SyszuxDetectYolo::SyszuxDetectYolo(std::string path, std::string device):Deepvac(path, device){
}

void SyszuxDetectYolo::set(int input_size, float iou_thresh, float score_thresh, std::vector<std::string> idx_to_cls) {
    input_size_ = input_size;
    iou_thresh_ = iou_thresh;
    score_thresh_ = score_thresh;
    idx_to_cls_ = idx_to_cls;
}

torch::Tensor SyszuxDetectYolo::postProcess(torch::Tensor& preds) {
    std::vector<torch::Tensor> index = torch::where(preds.select(2, 4) > score_thresh_);
    auto pred = preds.index_select(1, index[1]).squeeze();
    if (pred.sizes()[0] == 0) {
        return torch::zeros({0, 6});
    }

    pred.slice(1, 5, pred.sizes()[1]) = pred.slice(1, 5, pred.sizes()[1]) * pred.slice(1, 4, 5);
    pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2).div(2);
    pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3).div(2);
    pred.select(1, 2) = pred.select(1, 2) + pred.select(1, 0);
    pred.select(1, 3) = pred.select(1, 3) + pred.select(1, 1);
    
    std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
    
    auto max_conf = std::get<0>(max_classes).unsqueeze(1);
    auto max_index = std::get<1>(max_classes).unsqueeze(1);

    pred = torch::cat({pred.slice(1, 0, 4), max_conf, max_index}, 1);
    if (pred.sizes()[0] == 0) {
        return torch::zeros({0, 6});
    }

    pred = torch::index_select(pred, 0, torch::nonzero(max_conf.view(-1) > score_thresh_).select(1, 0));

    int max_side = 4096;
    auto class_offset = pred.slice(1, 5, 6) * max_side;
    auto boxes = pred.slice(1, 0, 4) + class_offset;
    auto scores = pred.slice(1, 4, 5);
    auto dets = torch::cat({boxes, scores}, 1);
    auto keep = gemfield_org::nms(dets, iou_thresh_);
    pred = pred.index(keep);

    return pred;

}

std::optional<std::pair<std::vector<std::string>, torch::Tensor>> SyszuxDetectYolo::process(cv::Mat& frame){
    cv::Mat input_img = frame.clone();
    
    int h = input_img.rows;
    int w = input_img.cols;
    float r = std::min((float)input_size_ / (float)h, (float)input_size_ / (float)w);
    
    int w_new = (int)(std::round((float)w * r));
    int h_new = (int)(std::round((float)h * r));
    
    float dw = (float)(input_size_ - w_new) / 2.0f;
    float dh = (float)(input_size_ - h_new) / 2.0f;

    if (h != h_new and w != w_new) {
        cv::resize(input_img, input_img, cv::Size(w_new, h_new), cv::INTER_LINEAR);
    }
    int top = ((int)(input_size_) - h_new) / 2;
    int bottom = ((int)(input_size_)- h_new + 1) / 2;
    int left = ((int)(input_size_)- w_new) / 2;
    int right = ((int)(input_size_)- w_new + 1) / 2;
    
    cv::copyMakeBorder(input_img, input_img, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
    
    auto input_tensor_opt = gemfield_org::cvMat2Tensor(std::move(input_img), gemfield_org::NORMALIZE0_1, gemfield_org::NO_MEAN_STD);

    if(!input_tensor_opt){
        return std::nullopt;
    }
    auto input_tensor = input_tensor_opt.value();

    auto output = forward<std::vector<c10::IValue>>(input_tensor);
    torch::Tensor preds = output[0].toTensor();
    
    auto pred = postProcess(preds);
    std::vector<std::string> classes;
    torch::Tensor scores;
    
    if (pred.sizes()[0] == 0) {
        classes.push_back("None");
        scores = torch::zeros({1});
        std::pair result(classes, scores);
        return result;
    }
    
    scores = pred.select(1, pred.sizes()[1]-2);
    auto classes_t = pred.select(1, pred.sizes()[1]-1);
 
    for (int i=0; i<pred.sizes()[0]; i++) {
        classes.push_back(idx_to_cls_[classes_t[i].item().toLong()]);
    }
    std::pair result(classes, scores);
    return result;
}
}//namespace
