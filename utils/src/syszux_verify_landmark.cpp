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
int decodeBox(torch::Tensor loc, torch::Tensor prior, torch::Tensor variances, torch::Tensor& boxes)
{
    torch::Tensor temp = torch::rand({loc.size(0), loc.size(1)});
    boxes = torch::cat({ torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(loc.slice(1, 0, 2), variances[0]), prior.slice(1, 2, 4))),
                    torch::mul(prior.slice(1, 2, 4), torch::exp(torch::mul(loc.slice(1, 2, 4), variances[1])))}, 1);
    boxes.slice(1, 0, 2) = torch::sub(boxes.slice(1, 0, 2), torch::div(boxes.slice(1, 2, 4), 2));
    boxes.slice(1, 2, 4) = torch::add(boxes.slice(1, 2, 4), boxes.slice(1, 0, 2));
    return 0;
}

int decodeLandmark(torch::Tensor prior, torch::Tensor variances, torch::Tensor& landms)
{
    landms = torch::cat({
                    torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(landms.slice(1, 0, 2), variances[0]), prior.slice(1, 2, 4))),
                    torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(landms.slice(1, 2, 4), variances[0]), prior.slice(1, 2, 4))),
                    torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(landms.slice(1, 4, 6), variances[0]), prior.slice(1, 2, 4))),
                    torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(landms.slice(1, 6, 8), variances[0]), prior.slice(1, 2, 4))),
                    torch::add(prior.slice(1, 0, 2), torch::mul(torch::mul(landms.slice(1, 8, 10), variances[0]), prior.slice(1, 2, 4)))
                    }, 1);
    return 0;
}
}//namespace
