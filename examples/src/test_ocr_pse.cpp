/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_ocr_pse.h"
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
    int long_size = 1280;
    int crop_gap = 5;
    SyszuxOcrPse ocr_detect(ocr_pse_deepvac, device);
    ocr_detect.set(long_size, crop_gap);

    auto mat_opt = gemfield_org::img2CvMat(path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto mat_out = mat_opt.value();
    auto detect_out_opt = ocr_detect.process(mat_out);
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
