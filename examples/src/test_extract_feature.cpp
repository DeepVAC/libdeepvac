/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include "deepvac_loader.h"
#include "test_extract_feature.h"
#include <torch/serialize.h>

namespace deepvac{

std::string getLabelFromPath(std::string& file_path, char seperator = '/')
{
    std::size_t sep_pos = file_path.rfind(seperator);
    if(sep_pos != std::string::npos){
        return file_path.substr(sep_pos + 1, file_path.size() -1);
    }
    return "unknown";
}

void FeatureEmbFromDir::dumpEmb(std::string output_feature_file){
    torch::NoGradGuard no_grad;
    for (auto& f_m : loader_){
        auto [f, m] = f_m;
        if(!m){
            continue;
        }
        {GEMFIELD_DI2("input file: ", f.c_str());}
        at::Tensor emb = deepvac_(*m);
        emb_vec_.push_back(emb);

        std::string base_dir = std::filesystem::path(f).parent_path();
        name_vec_.push_back(getLabelFromPath(base_dir));
    }

    feature_ = torch::cat(emb_vec_, 0);

    std::stringstream feature_size;
    feature_size << feature_.sizes();
    std::string msg = gemfield_org::format("%s: %s | %s : %d", "load feature size: ", feature_size.str().c_str(), "load name vector size: ", name_vec_.size() );
    GEMFIELD_I(msg.c_str());

    std::ofstream name_output(output_feature_file + ".name");
    for (auto &name : name_vec_){
        name_output << name << "\n";
    }
    torch::save(feature_ , output_feature_file);
}

void FeatureEmbFromDir::operator()(std::string input_feature_file){
    torch::NoGradGuard no_grad;
    std::ifstream inputstream(input_feature_file + ".name");

    std::copy(std::istream_iterator<std::string>(inputstream),
                std::istream_iterator<std::string>(),back_inserter(name_vec_));
    
    torch::load(feature_,input_feature_file);
    feature_ = feature_.to(deepvac_.getDevice());

    std::stringstream feature_size;
    feature_size << feature_.sizes();
    std::string msg = gemfield_org::format("%s: %s | %s : %d", "load feature size: ", feature_size.str().c_str(), "load name vector size: ", name_vec_.size() );
    GEMFIELD_I(msg.c_str());

    for (auto& f_m : loader_){
        auto [f, m] = f_m;
        if(!m){
            continue;
        }

        {GEMFIELD_DI2("input file: ", f.c_str());}
        at::Tensor emb = deepvac_(*m);
        at::Tensor distance = torch::norm(feature_ - emb, 2,  1);
        int min_index = torch::argmin(distance).item<int64_t>();
        GEMFIELD_DI2("predict: ", name_vec_.at(min_index).c_str());
    }
}

} //namespace deepvac

using namespace deepvac;
int main(int argc, const char* argv[]) {
    if (argc != 4) {
        GEMFIELD_E("usage: deepvac <dumpEmb|predict> <path-to-exported-torchscript-module> <img_path1>");
        return -1;
    }

    FeatureEmbFromDir emb_fromdir( Deepvac(argv[2], "cuda:1"), argv[3]);
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