/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_face_detect.h"
#include "syszux_img2tensor.h"

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];
    SyszuxFaceDetect face_detect(device);
    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto mat_out = mat_opt.value();
    auto detect_out_opt = face_detect(mat_out);
    if(!detect_out_opt){
        throw std::runtime_error("no face detected");
    }
    auto detect_out = detect_out_opt.value();
    for (int i=0; i<detect_out.size(); i++){
        cv::imwrite("./test_detect_face_"+std::to_string(i)+".jpg", detect_out[i]);
    }
    return 0;
}
