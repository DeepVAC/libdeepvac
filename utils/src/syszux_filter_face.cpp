/*
 *  Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 *  This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 *  You may not use this file except in compliance with the License.
 */

#include <iostream>
#include "opencv2/opencv.hpp"
#include "syszux_filter_face.h"

namespace gemfield_org{
bool filterFace(cv::Mat img, std::vector<float> bbox, std::vector<float> landmark){
    cv::Mat tempImg = img(cv::Rect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]));
    if(tempImg.cols<100 || tempImg.rows<100){
        return false;
    }
    std::tuple<float, float> leye(landmark[0], landmark[1]);
    std::tuple<float, float> reye(landmark[2], landmark[3]);
    std::tuple<float, float> nose(landmark[4], landmark[5]);
    std::tuple<float, float> lmouth(landmark[6], landmark[7]);
    std::tuple<float, float> rmouth(landmark[8], landmark[9]);

    auto lx = std::get<0>(nose) - std::get<0>(leye);
    if(lx <= 0){
        return false;
    }
    auto rx = std::get<0>(reye) - std::get<0>(nose);
    if(rx <= 0){
        return false;
    }
    auto eyex = lx / rx;
    auto mx = std::get<0>(nose) - std::get<0>(lmouth);
    if(mx <= 0){
        return false;
    }
    auto mx2 = std::get<0>(rmouth) - std::get<0>(nose);
    if(mx2 <= 0){
        return false;
    }
    auto mouthx = mx / mx2;
    if((eyex<0.6 || eyex>1.4) || (mouthx<0.5 || mouthx>3)){
        return false;
    }
    return true;
}
}//namespace
