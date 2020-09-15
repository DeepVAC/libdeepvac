/*
 *  Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 *  This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 *  You may not use this file except in compliance with the License.
 */

#include "syszux_verify_landmark.h"

namespace gemfield_org{
bool isValidLandmark(std::vector<float> bbox, std::vector<float> landmark, int min_face_size){
    auto width = bbox[2]-bbox[0];
    auto height = bbox[3]-bbox[1];
    if (width < min_face_size || height < min_face_size){
        return false;
    }
    // landmark x1,y1,x2,y2,...,x5,y5
    auto leye_x = landmark[0];
    auto reye_x = landmark[2];
    auto nose_x = landmark[4];
    auto lmouth_x = landmark[6];
    auto rmouth_x = landmark[8];

    auto lx_eye = nose_x - leye_x;
    if(lx_eye <= 0){
        return false;
    }
    auto rx_eye = reye_x - nose_x;
    if(rx_eye <= 0){
        return false;
    }
    auto eyex = lx_eye / rx_eye;
    auto lx_mouth = nose_x - lmouth_x;
    if(lx_mouth <= 0){
        return false;
    }
    auto rx_mouth = rmouth_x - nose_x;
    if(rx_mouth <= 0){
        return false;
    }
    auto mouthx = lx_mouth / rx_mouth;
    if((eyex<0.6 || eyex>1.4) || (mouthx<0.5 || mouthx>3)){
        return false;
    }
    return true;
}
}//namespace
