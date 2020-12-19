/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_cls_resnet.h"
#include "syszux_img2tensor.h"

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string img_path = argv[2];
    
    std::vector<int> input_size = {224, 224};
    SyszuxClsResnet cls_resnet(cls_resnet_deepvac, device);
    cls_resnet.set(input_size);
    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected!");
        return 1;
    }
    auto mat_out = mat_opt.value();
    
    auto resnet_out_opt = cls_resnet.process(mat_out);
    if(!resnet_out_opt){
	throw std::runtime_error("return empty error!");
    }
    auto resnet_out = resnet_out_opt.value();
    std::cout << "Index: " << resnet_out.first << std::endl;
    std::cout << "Probability: " << resnet_out.second << std::endl;
    return 0;
}
