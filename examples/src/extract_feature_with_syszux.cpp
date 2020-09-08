/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "deepvac_feature.h"
#include "deepvac_loader.h"
#include "syszux_decrypt.h"

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 4) {
        GEMFIELD_E("usage: deepvac <dumpEmb|predict> <path-to-exported-torchscript-module> <img_path1>");
        return -1;
    }

    auto de = gemfield_org::SyszuxDecrypt("gemfieldisacivilnetmaintainer");

    FeatureEmbFromDir emb_fromdir( Deepvac(de.de(argv[2]), "cuda:1"), argv[3]);
    std::string feature_file = "gemfield_org.feature";
    std::string op = argv[1];

    if(op == "dumpEmb"){
        emb_fromdir.dumpEmb(feature_file);
        return 0;
    }

    if(op == "predict"){
        emb_fromdir(feature_file);
        return 0;
    }

    GEMFIELD_E("usage: deepvac <dumpEmb|predict> <path-to-exported-torchscript-module> <img_path1>");
    return -1;
}