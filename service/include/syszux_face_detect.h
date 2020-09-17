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
class SyszuxFaceDetect : public Deepvac{
    public:
        SyszuxFaceDetect(std::string device = "cpu");
        SyszuxFaceDetect(const SyszuxFaceDetect&) = delete;
        SyszuxFaceDetect& operator=(const SyszuxFaceDetect&) = delete;
        SyszuxFaceDetect(SyszuxFaceDetect&&) = default;
        SyszuxFaceDetect& operator=(SyszuxFaceDetect&&) = default;
        virtual ~SyszuxFaceDetect() = default;
        virtual std::optional<std::vector<cv::Mat>> operator() (cv::Mat frame);
    
    private:
        gemfield_org::PriorBox prior_box_;
        gemfield_org::AlignFace align_face_;
};
} //namespace deepvac
