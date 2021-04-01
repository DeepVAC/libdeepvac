/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "syszux_face_retina_config.h"
#include "gemfield.h"

namespace deepvac{


void SyszuxFaceRetinaConfig::initParameter(std::string device){
    variances_tensor_ = torch::tensor({0.1, 0.2}).to(device);
    prior_box_ = gemfield_org::PriorBox({{16,32},{64,128},{256,512}}, {8,16,32});
    setTopK(50);
    setKeepTopK(50);
    setConfThreshold(0.4);
    setNMSThreshold(0.4);
    setMaxHW(2000);
    setGapThreshold(0.1);

    last_w_ = 0;
    last_h_ = 0;
    last_prior_ = torch::ones({1, 4});
    last_box_scale_ = torch::ones({1, 4});
    last_lmk_scale_ = torch::ones({1, 10});
}

void SyszuxFaceRetinaConfig::setTopK(int top_k){
    top_k_ = top_k;
}

void SyszuxFaceRetinaConfig::setKeepTopK(int keep_top_k){
    keep_top_k_ = keep_top_k;
}

void SyszuxFaceRetinaConfig::setConfThreshold(float confidence_threshold){
    confidence_threshold_ = confidence_threshold;
}

void SyszuxFaceRetinaConfig::setNMSThreshold(float nms_threshold){
    nms_threshold_ = nms_threshold;
}

void SyszuxFaceRetinaConfig::setMaxHW(int max_hw){
    max_hw_ = max_hw;
}

void SyszuxFaceRetinaConfig::setGapThreshold(float gap_threshold){
    gap_threshold_ = gap_threshold;
}

} //namespace deepvac
