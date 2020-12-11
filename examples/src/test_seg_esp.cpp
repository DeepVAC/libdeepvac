/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_seg_esp.h"
#include "syszux_img2tensor.h"

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];
    std::vector<int> image_size = {512, 256};
    SyszuxSegEsp seg(seg_esp_deepvac, device);
    seg.set(image_size);
    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 0;
    }
    auto mat_out = mat_opt.value();
    auto seg_out_opt = seg.process(mat_out);
    if(!seg_out_opt){
        throw std::runtime_error("seg error!");
	return 0;
    }
    auto seg_out = seg_out_opt.value();

    cv::Mat seg_img = mat_out.clone();
    for (int i = 0; i < seg_out.rows; ++i) {
        for (int j = 0; j < seg_out.cols; ++j) {
            if ((int)(seg_out.at<uchar>(i, j)) == 1) {
                *(seg_img.data + seg_img.step[0] * i + seg_img.step[1] * j + seg_img.elemSize1() * 0) = 0;
                *(seg_img.data + seg_img.step[0] * i + seg_img.step[1] * j + seg_img.elemSize1() * 1) = 0;
                *(seg_img.data + seg_img.step[0] * i + seg_img.step[1] * j + seg_img.elemSize1() * 2) = 255;
            }
        }
    }
    cv::imwrite("./res.jpg", seg_img);
    return 0;
}
