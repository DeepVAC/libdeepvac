/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include "syszux_priorbox.h"
#include "syszux_nms.h"
#include "syszux_align_face.h"
#include "syszux_verify_landmark.h"
#include "syszux_img2tensor.h"

namespace deepvac{
class SyszuxFaceRetinaConfig {
    public:
        SyszuxFaceRetinaConfig() = default;
        SyszuxFaceRetinaConfig(const SyszuxFaceRetinaConfig&) = default;
        SyszuxFaceRetinaConfig& operator=(const SyszuxFaceRetinaConfig&) = default;
        SyszuxFaceRetinaConfig(SyszuxFaceRetinaConfig&&) = default;
        SyszuxFaceRetinaConfig& operator=(SyszuxFaceRetinaConfig&&) = default;
        virtual ~SyszuxFaceRetinaConfig() = default;
        void initParameter(std::string device);
        void setTopK(int top_k);
        void setKeepTopK(int keep_top_k);
        void setConfThreshold(float confidence_threshold);
        void setNMSThreshold(float nms_threshold);
        void setMaxHW(int max_hw);
        void setGapThreshold(float gap_threshold);

    protected:
        gemfield_org::PriorBox prior_box_;
        gemfield_org::AlignFace align_face_;
        int top_k_;
        int keep_top_k_;
        float nms_threshold_;
        float confidence_threshold_;
        int max_hw_;
        torch::Tensor variances_tensor_;
        int last_h_;
        int last_w_;
        torch::Tensor last_prior_;
        torch::Tensor last_box_scale_;
        torch::Tensor last_lmk_scale_;
        float gap_threshold_;
};
} //namespace deepvac
