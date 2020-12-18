/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_priorbox.h"
#include "syszux_img2tensor.h"

int main(int argc, const char* argv[]) {
    gemfield_org::PriorBox pb({{16,32},{64,128},{256,512}}, {8,16,32});
    std::vector<int> img_size = {224,312};
    auto x = pb.forward(img_size);
    std::cout<<x<<std::endl;
    return 0;
}