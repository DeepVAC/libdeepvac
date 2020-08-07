/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include <opencv2/core/core.hpp>
#include <torch/script.h>
#include <string>
#include <memory>
#include "gemfield.h"

#define SYSZUX_EXPORT __attribute__((__visibility__("default")))
#define SYSZUX_HIDDEN __attribute__((__visibility__("hidden")))

namespace deepvac {
//from pytorch 1.2, script::Module is now a reference type
class SYSZUX_EXPORT Deepvac : public std::enable_shared_from_this<Deepvac> {
    public:
        Deepvac() = delete;
        Deepvac(const Deepvac&) = delete;
        Deepvac& operator=(const Deepvac&) = delete;
        Deepvac(Deepvac&&) = default;
        Deepvac& operator=(Deepvac&&) = default;
        virtual ~Deepvac() = default;
        explicit Deepvac(const char* model_path, c10::optional<c10::Device> device = c10::nullopt);
        explicit Deepvac(std::string model_path, c10::optional<c10::Device> device = c10::nullopt):Deepvac(model_path.c_str(), device){}
        virtual std::vector<at::Tensor> operator() (cv::Mat& frame);

    private:
        virtual std::vector<at::Tensor> getEmbFromCvMat(cv::Mat& frame);
        std::unique_ptr<torch::jit::script::Module> module_;
};

}// namespace deepvac