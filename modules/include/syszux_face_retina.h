/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include "deepvac.h"
#include "syszux_priorbox.h"
#include "syszux_nms.h"
#include "syszux_align_face.h"
#include "syszux_verify_landmark.h"
#include "syszux_img2tensor.h"

namespace deepvac{
class SyszuxFaceRetina : public Deepvac{
    public:
        SyszuxFaceRetina(std::string path, std::string device = "cpu");
        SyszuxFaceRetina(const SyszuxFaceRetina&) = delete;
        SyszuxFaceRetina& operator=(const SyszuxFaceRetina&) = delete;
        SyszuxFaceRetina(SyszuxFaceRetina&&) = default;
        SyszuxFaceRetina& operator=(SyszuxFaceRetina&&) = default;
        virtual ~SyszuxFaceRetina() = default;
    public:
        std::optional<std::vector<cv::Mat>> process(cv::Mat frame);
    
    private:
        gemfield_org::PriorBox prior_box_;
        gemfield_org::AlignFace align_face_;
};
} //namespace deepvac
