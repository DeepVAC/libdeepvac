/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "gemfield.h"
#include "syszux_cls_mobile.h"
#include "syszux_img2tensor.h"

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

    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto mat_out = mat_opt.value();

    auto cls_out_opt = cls.process(mat_out);
    if(!cls_out_opt) {
        return 0;
    }
    auto cls_out = cls_out_opt.value();
    std::cout << "Index: " << cls_out.first << std::endl;
    std::cout << "Probability: " << cls_out.second << std::endl;
    return 0;
}
