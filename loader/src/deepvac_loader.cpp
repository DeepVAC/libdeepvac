/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include <algorithm>
#include <cctype>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "deepvac_loader.h"
#include "gemfield.h"
#include "syszux_img2tensor.h"

namespace deepvac{

const std::string RecursiveFileIter::operator*() const {
    std::string p = (*walker_).path(); 
    return p;
}

const std::optional<cv::Mat> CvMatIter::operator*() const {
    std::string p = (*walker_).path();
    return gemfield_org::img2CvMat(p);
}

const std::tuple<std::string, std::optional<at::Tensor>> ImgFileInputTensorPairIter::operator*() const {
    std::string p = (*walker_).path();
    std::string suffix = std::filesystem::path(p).extension();

    if (suffix.empty()){
        return {p, std::nullopt};
    }

    std::transform(suffix.begin(), suffix.end(), suffix.begin(),[](unsigned char c){ return std::tolower(c); });

    if(suffix_.find(suffix) == suffix_.end() ){
        return {p, std::nullopt};
    }

    return {p, gemfield_org::img2tensor(p)};
}

//loader that based on std container.
std::optional<cv::Mat> DeepvacVectorIter::operator*() const {
    return gemfield_org::img2CvMat(*walker_);
}

} //namespace deepvac