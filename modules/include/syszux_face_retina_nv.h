/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <tuple>
#include <vector>
#include "syszux_img2tensor.h"
#include "syszux_tensorrt_buffers.h"
#include "deepvac_nv.h"
#include "syszux_face_retina_config.h"

namespace deepvac{
class SyszuxFaceRetinaNV : public DeepvacNV, public SyszuxFaceRetinaConfig {
    public:
        SyszuxFaceRetinaNV(std::string path, std::string device = "cpu");
        SyszuxFaceRetinaNV(std::vector<unsigned char>&& buffer, std::string device = "cpu");
        SyszuxFaceRetinaNV(const SyszuxFaceRetinaNV&) = delete;
        SyszuxFaceRetinaNV& operator=(const SyszuxFaceRetinaNV&) = delete;
        SyszuxFaceRetinaNV(SyszuxFaceRetinaNV&&) = default;
        SyszuxFaceRetinaNV& operator=(SyszuxFaceRetinaNV&&) = default;
        virtual ~SyszuxFaceRetinaNV() = default;
        virtual std::optional<std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>>> process(cv::Mat frame);
};

} //namespace deepvac
