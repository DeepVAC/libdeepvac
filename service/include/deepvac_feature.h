/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "deepvac_loader.h"
#include "deepvac.h"

namespace deepvac{

template <typename DeepvacLoaderType>
class FeatureEmbBase{
    public:
        FeatureEmbBase() = delete;
        FeatureEmbBase(const FeatureEmbBase&) = delete;
        FeatureEmbBase& operator=(const FeatureEmbBase&) = delete;
        FeatureEmbBase(FeatureEmbBase&&) = default;
        FeatureEmbBase& operator=(FeatureEmbBase&&) = default;
        virtual ~FeatureEmbBase() = default;
        FeatureEmbBase(Deepvac&& deepvac, DeepvacLoaderType loader):
            loader_(std::move(loader)), deepvac_(std::move(deepvac)){}
        virtual void operator() () {};
    
    protected:
        DeepvacLoaderType loader_;
        Deepvac deepvac_;
};

using pairloader = DeepvacRecursiveFileLoader<ImgFileCvMatPairIter>;
class FeatureEmbFromDir : public FeatureEmbBase<pairloader>{
    public:
        FeatureEmbFromDir(Deepvac&& deepvac, const char* url = ""): 
                FeatureEmbBase<pairloader>(std::move(deepvac), pairloader(url)) {}

        virtual void operator() ();
};

// class FeatureEmbWriter : public FeatureEmbPredict{

// };

} //namespace deepvac