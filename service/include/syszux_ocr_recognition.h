/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include "syszux_adaptor.h"
#include <torch/script.h>

//namespace gemfield_org {
class SyszuxOcrRecognition{
    public:
        SyszuxOcrRecognition(std::string device="cpu");
        void operator() (cv::Mat img);
    private:
        std::string device_;
};
//} //namespace
