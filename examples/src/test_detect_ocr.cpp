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
    int crop_gap = 10;
    SyszuxOcrDetect ocr_detect(device);
    ocr_detect.set(long_size, crop_gap);
   
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

    std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> detect_out = detect_out_opt.value();
    std::vector<cv::Mat> crop_imgs = detect_out.first;
    std::vector<std::vector<int>> rects = detect_out.second;

    if (crop_imgs.size()==0) {
        std::cout << "no text detected" << std::endl;
        return 0;
    }
    for (int i=0; i<crop_imgs.size(); i++) {
        cv::imwrite("./ocr_detect_test" + std::to_string(i) + ".jpg", crop_imgs[i]);
        std::cout << "rect: " << rects[i] << std::endl;
    }
    return 0;
}
