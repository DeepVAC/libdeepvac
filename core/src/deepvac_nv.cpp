/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include <type_traits>
#include <chrono>
#include <ctime> 
#include <iostream>
#include <cassert>
#include "deepvac_nv.h"

namespace deepvac {
DeepvacNV::DeepvacNV(const char* path, std::string device){
    GEMFIELD_SI;
    auto start = std::chrono::system_clock::now();
    try{
        device_ = device;
        setModel(path);
    }catch(...){
        std::string msg =  "Internal ERROR!";
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("NV Model loading time: %f", model_loading_duration.count());
    GEMFIELD_DI(msg.c_str());
}

DeepvacNV::DeepvacNV(std::vector<unsigned char>&& buffer, std::string device){
    GEMFIELD_SI;
    auto start = std::chrono::system_clock::now();
    try{
        device_ = device;
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
        trt_module_ = makeUnique(runtime->deserializeCudaEngine((void*)buffer.data(), buffer.size(), nullptr));
        assert(trt_module_ != nullptr);
        trt_context_ = makeUnique(trt_module_->createExecutionContext());
        assert(trt_context_ != nullptr);
        runtime->destroy();
        runtime = nullptr;
    }catch(...){
        std::string msg =  "Internal ERROR!";
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
    }
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("NV Model loading time: %f", model_loading_duration.count());
    GEMFIELD_DI(msg.c_str());
}

void DeepvacNV::setBinding(int io_num) {
    for(int i = 0; i < io_num; ++i) {
        datas_.emplace_back(gemfield_org::ManagedBuffer());
    }
}

void DeepvacNV::setModel(const char* model_path) {
    std::ifstream in(model_path, std::ifstream::binary);
    if(in.is_open()) {
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);
        //initLibNvInferPlugins(&logger_, "");
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
        trt_module_ = makeUnique(runtime->deserializeCudaEngine((void*)engineBuf.get(), bufCount, nullptr));
        assert(trt_module_ != nullptr);
        trt_context_ = makeUnique(trt_module_->createExecutionContext());
        assert(trt_context_ != nullptr);
        //mBatchSize = trt_module_->getMaxBatchSize();
        //spdlog::info("max batch size of deserialized engine: {}",mEngine->getMaxBatchSize());
        runtime->destroy();
        runtime = nullptr;
    }
}

std::vector<void*> DeepvacNV::prepareInputOutput(const at::Tensor& input) {
    auto sizes = input.sizes();
    nvinfer1::Dims dim;
    dim.nbDims = sizes.size();
    std::copy(sizes.begin(), sizes.end(), dim.d);
    trt_context_->setBindingDimensions(0, dim);
    auto nb_bind = trt_module_->getNbBindings();
    //input
    datas_[0].deviceBuffer.fromTensor(input);
    //output
    for(auto i = 1; i < nb_bind; ++i) {
        auto dims = trt_context_->getBindingDimensions(i);
        std::vector<int64_t> shape;
        std::copy(dims.d, dims.d+dims.nbDims, std::back_inserter(shape));
        datas_[i].deviceBuffer.resize(shape);
    }
    std::vector<void*> predicitonBindings;
    for(auto i = 0; i < nb_bind; ++i) {
        predicitonBindings.emplace_back(datas_[i].deviceBuffer.data());
    }
    return predicitonBindings;
}

} //namespace deepvac
