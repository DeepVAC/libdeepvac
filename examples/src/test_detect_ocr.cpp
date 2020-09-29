/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_ocr_detect.h"
#include "gemfield.h"

using namespace deepvac;
int main(int argc, char** argv)
{
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }
    std::string device = argv[1];
    std::string path = argv[2];
    int long_size = 640;
    SyszuxOcrDetect ocr_detect(long_size, device);
   
    cv::Mat img_raw = cv::imread(path);
    if(img_raw.data == nullptr)
    {
        std::cerr<< path << " is not a image file!" << std::endl;
        return 0;
    }
    auto detect_out_opt = ocr_detect(img_raw);
    if(!detect_out_opt){
        throw std::runtime_error("no text detected");
    }
    cv::Mat detect_out = detect_out_opt.value();	
    cv::imwrite("./ocr_detect_test.jpg", detect_out);
    return 0;
}
