/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "gemfield.h"
#include "syszux_cls_mobile.h"

using namespace deepvac;
int main(int argc, char** argv)
{
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];

    SyszuxClsMobile cls(cls_mobile_deepvac, device);

    cv::Mat img_raw = cv::imread(img_path);
    if(img_raw.data == nullptr){
        GEMFIELD_E(path + " is not a image file! Please input a image path...");
        return -1;
    }
    auto cls_out_opt = cls.process(img_raw);
    if(!cls_out_opt) {
        return 0;
    }
    auto cls_out = cls_out_opt.value();
    std::cout << "Index: " << cls_out.first << std::endl;
    std::cout << "Probability: " << cls_out.second << std::endl;
    return 0;
}
