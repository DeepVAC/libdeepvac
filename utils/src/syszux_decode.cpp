/*
 * * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * * You may not use this file except in compliance with the License.
 * */

#include "syszux_decode.h"

namespace gemfield_org{
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
