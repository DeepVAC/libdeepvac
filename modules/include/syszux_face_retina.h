/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <tuple>

#include "deepvac.h"
#include "syszux_priorbox.h"
#include "syszux_nms.h"
#include "syszux_align_face.h"
#include "syszux_verify_landmark.h"
#include "syszux_img2tensor.h"
#include "syszux_face_retina_config.h"

namespace deepvac{
class SyszuxFaceRetina : public Deepvac, public SyszuxFaceRetinaConfig {
    public:
        SyszuxFaceRetina() = default;
        SyszuxFaceRetina(std::string path, std::string device = "cpu");
        SyszuxFaceRetina(std::vector<unsigned char>&& buffer, std::string device = "cpu");
        SyszuxFaceRetina(const SyszuxFaceRetina&) = default;
        SyszuxFaceRetina& operator=(const SyszuxFaceRetina&) = default;
        SyszuxFaceRetina(SyszuxFaceRetina&&) = default;
        SyszuxFaceRetina& operator=(SyszuxFaceRetina&&) = default;
        virtual ~SyszuxFaceRetina() = default;
        std::optional<std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>>> process(cv::Mat frame);
};
} //namespace deepvac
