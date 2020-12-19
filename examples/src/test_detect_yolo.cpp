/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "gemfield.h"
#include "syszux_detect_yolo.h"

using namespace deepvac;
int main(int argc, char** argv)
{
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];

    int input_size = 512;
    float iou_thresh = 0.45;
    float score_thresh = 0.25;
    std::vector<std::string> idx_to_cls = {"1", "2", "3", "4"};
    SyszuxDetectYolo detect(detect_yolo_deepvac, device);
    detect.set(input_size, iou_thresh, score_thresh);

    cv::Mat img_raw = cv::imread(img_path);
    if(img_raw.data == nullptr){
        GEMFIELD_E(path + " is not a image file! Please input a image path...");
        return -1;
    }
    auto detect_out_opt = detect.process(img_raw);
    if(!detect_out_opt) {
        return 0;
    }
    auto detect_out = detect_out_opt.value();
    
    std::cout << "size: " << detect_out.size() << std::endl;
    if (detect_out.size()==0) {
        std::cout << "detect none."  << std::endl;
        return 0;
    }

    for (int j=0; j<detect_out.size(); j++) {
        int idx = detect_out[j].first;
        std::vector<float> bbox_and_score = detect_out[j].second;
        std::cout << "class: " << idx_to_cls[idx] << std::endl;
        std::cout << "bbox_and_score: " << bbox_and_score << std::endl;
    }
    return 0;
}
