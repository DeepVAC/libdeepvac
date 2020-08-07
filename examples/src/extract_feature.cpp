/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "deepvac_feature.h"
#include "deepvac_loader.h"

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <path-to-exported-torchscript-module> <img_path1>");
        return -1;
    }

    FeatureEmbFromDir emb_fromdir( Deepvac(argv[1], torch::kCUDA), argv[2]);
    emb_fromdir();
}