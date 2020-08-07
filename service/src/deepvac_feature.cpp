/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "deepvac_feature.h"

namespace deepvac{

void FeatureEmbFromDir::operator() (){
    for (auto& f_m : loader_){
        auto [f, m] = f_m;
        if(!m){
            continue;
        }
        std::vector<at::Tensor> rc = deepvac_(*m);
        std::cout<<"result: " <<f<<"\t"<< rc.size() <<std::endl;
    }
}
    
} //namespace deepvac