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
    SyszuxFaceDetect face_detect( Deepvac("/home/gemfield/detect.gemfield", device));
    auto tensor_out_opt = gemfield_org::img2CvMat(img_path);
    if(!tensor_out_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto tensor_out = tensor_out_opt.value();
    auto detect_out = face_detect(tensor_out);

    return 0;
}