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
    std::vector<std::string> idx_to_cls = {"Female-Breast", "Female-Genitals", "Male-Genitals", "Buttocks"};
    SyszuxDetectYolo detect("/gemfield/hostpv/wangyuhang/yolov5/weights/last_1.torchscript.pt", device);
    detect.set(input_size, iou_thresh, score_thresh, idx_to_cls);

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
    auto classes = detect_out.first;
    auto scores = detect_out.second;

    for (int j=0; j<classes.size(); j++) {
        if (classes[j]=="None") {
            std::cout << "detect none."  << std::endl;
            break;
        }
        std::cout << "class: " << classes[j] << std::endl;
        std::cout << "score: " << scores[j] << std::endl;
    }
    return 0;
}
