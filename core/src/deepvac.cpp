/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include <type_traits>
#include <filesystem>
#include <chrono>
#include <ctime> 
#include <iostream>
#include "deepvac.h"
#include "syszux_stream_buffer.h"

namespace deepvac {

Deepvac::Deepvac(const char* model_path, std::string device){
    GEMFIELD_SI;
    auto start = std::chrono::system_clock::now();
    try{
        device_ = device;
        module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, device_));
    }catch(const c10::Error& e){
        std::string msg = gemfield_org::format("%s: %s", "ERROR MODEL: ",e.what_without_backtrace() );
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }catch(...){
        std::string msg =  "Internal ERROR!";
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("Model loading time: %f", model_loading_duration.count());
    GEMFIELD_DI(msg.c_str());
    module_->eval();
}

Deepvac::Deepvac(std::vector<unsigned char>&& buffer, std::string device){
    GEMFIELD_SI;
    auto start = std::chrono::system_clock::now();
    try{
        SyszuxStreamBuffer databuf(buffer.data(), buffer.size());
        std::istream is(&databuf);
        device_ = device;
        module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(is, device_));
    }catch(const c10::Error& e){
        std::string msg = gemfield_org::format("%s: %s", "ERROR MODEL: ",e.what_without_backtrace() );
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }catch(...){
        std::string msg =  "Internal ERROR!";
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("Model loading time: %f", model_loading_duration.count());
    GEMFIELD_DI(msg.c_str());
    module_->eval();
}

template<typename T> struct dependent_false : std::false_type {};
template<typename T = at::Tensor>
T Deepvac::forward(at::Tensor& t){
    GEMFIELD_SI;
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(t.to(device_));
    auto start = std::chrono::system_clock::now();
    T output;
    if constexpr (std::is_same_v<T, at::Tensor>){
        output = module_->forward(inputs).toTensor();
    }else if constexpr (std::is_same_v<T, std::vector<c10::IValue> >){
        output = module_->forward(inputs).toTuple()->elements();
    }else{
        static_assert(dependent_false<T>::value, "[libdeepvac] invalid usage of forward.");
    }
    std::chrono::duration<double> forward_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("forward time: %f",  forward_duration.count() );
    GEMFIELD_DI(msg.c_str());
    return output;
}

template at::Tensor Deepvac::forward<at::Tensor>(at::Tensor& t);
template std::vector<c10::IValue> Deepvac::forward<std::vector<c10::IValue>>(at::Tensor& t);

} //namespace deepvac
