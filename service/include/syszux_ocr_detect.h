/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

//#include "deepvac.h"
#include "syszux_adaptor.h"
#include <torch/script.h>

//namespace gemfield_org {
class SyszuxOcrDetect{
    public:
        SyszuxOcrDetect(std::string device);
        std::optional<cv::Mat> operator() (cv::Mat img, int long_size);
    private:
        std::string device_;
};
//} //namespace
