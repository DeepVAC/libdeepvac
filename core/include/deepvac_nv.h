/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once
#include <memory>
#include <vector>
#include <string>
#include "gemfield.h"
#include "syszux_tensorrt_buffers.h"
#include "NvInfer.h"

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override{
        // suppress info-level messages
        if (severity != Severity::kVERBOSE){
            std::cout << msg << std::endl;
        }
    }
};

struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj){
            obj->destroy();
        }
    }
};

namespace deepvac {
class DeepvacNV{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

    public:
        DeepvacNV() = default;
        explicit DeepvacNV(const char* model_path, std::string device);
        explicit DeepvacNV(std::string model_path, std::string device):DeepvacNV(model_path.c_str(), device){}
        explicit DeepvacNV(std::vector<unsigned char>&& buffer, std::string device);
        DeepvacNV(const DeepvacNV& rhs) = delete;
        DeepvacNV& operator=(const DeepvacNV& rhs) = delete;
        DeepvacNV(DeepvacNV&&) = default;
        DeepvacNV& operator=(DeepvacNV&&) = default;
        virtual ~DeepvacNV() = default;

    public:
        void setDevice(std::string device){device_ = device;}
        void setModel(const char* model_path);
        virtual void setBinding(int io_num);
        void** forward(void** data) {
            bool s = trt_context_->executeV2(data);
            return data;
        }

    protected:
        //all data members must be movable !!
        //all data members need dynamic memory must be managed by smart ptr !!
        SampleUniquePtr<nvinfer1::ICudaEngine> trt_module_;
        SampleUniquePtr<nvinfer1::IExecutionContext> trt_context_;
        template <typename T>
        SampleUniquePtr<T> makeUnique(T* t){
            return SampleUniquePtr<T>{t};
        }
        TrtLogger logger_;
        std::vector<gemfield_org::ManagedBuffer> datas_;
        std::string device_;
};

}// namespace deepvac
