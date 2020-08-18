/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
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
        explicit Deepvac(const char* model_path, std::string device = "cuda:0");
        explicit Deepvac(std::string model_path, std::string device = "cuda:0"):Deepvac(model_path.c_str(), device){}
        virtual at::Tensor operator() (at::Tensor& t);
        std::string getDevice(){return device_;}

    private:
        std::string device_;
        at::Tensor getEmbFromInputTensor(at::Tensor& t);
        std::unique_ptr<torch::jit::script::Module> module_;
};

}// namespace deepvac