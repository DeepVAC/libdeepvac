/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include <iostream>
#include <map>
#include "gemfield.h"
#include "syszux_cls_mobile.h"
#include "syszux_ocr_pse.h"

using namespace deepvac;
int main(int argc, char** argv)
{
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];

    int long_size = 1280;
    int crop_gap = 10;
    SyszuxOcrPse ocr_detect(ocr_pse_deepvac, device);
    ocr_detect.set(long_size, crop_gap);

    float confidence = 0.99;
    float threshold = 0.66;
    SyszuxClsMobile cls(cls_mobile_deepvac, device);
    cls.set(confidence, threshold);

    cv::Mat img_raw = cv::imread(img_path);
    if(img_raw.data == nullptr){
        GEMFIELD_E(path + " is not a image file! Please input a image path...");
        return -1;
    }
    //begin
    std::vector<std::pair<std::string, std::vector<int>>> result;
    auto detect_out_opt = ocr_detect.process(img_raw);
    if(!detect_out_opt){
        return 0;
    }
    std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> detect_out = detect_out_opt.value();
    std::vector<cv::Mat> crop_imgs = detect_out.first;
    std::vector<std::vector<int>> rects = detect_out.second;

    if (crop_imgs.size()==0) {
        return 0;
    }

    auto cls_out_opt = cls.process(crop_imgs);
    if(!cls_out_opt) {
        return 0;
    }
    auto cls_out = cls_out_opt.value();
    for (int i=0; i<cls_out.size(); i++) {
        cv::imwrite("res_"+std::to_string(i)+".jpg", cls_out[i]);
    }
    return 0;
}
