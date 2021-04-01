#pragma once

#include <cuda_runtime_api.h>
#include <numeric>
#include "NvInfer.h"


namespace gemfield_org{

inline unsigned int getElementSize(nvinfer1::DataType t){
    switch (t){
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d){
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
    public:
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : size_(0), capacity_(0), type_(type), buffer_(nullptr){}

        GenericBuffer(size_t size, nvinfer1::DataType type): size_(size), capacity_(size), type_(type){
            if (!alloc_fn_(&buffer_, this->nbBytes())){
                throw std::bad_alloc();
            }
        }

        GenericBuffer(GenericBuffer&& buf)
            : size_(buf.size_), capacity_(buf.capacity_), type_(buf.type_), buffer_(buf.buffer_){
            buf.size_ = 0;
            buf.capacity_ = 0;
            buf.type_ = nvinfer1::DataType::kFLOAT;
            buf.buffer_ = nullptr;
        }

        GenericBuffer& operator=(GenericBuffer&& buf){
            if (this != &buf){
                free_fn_(buffer_);
                size_ = buf.size_;
                capacity_ = buf.capacity_;
                type_ = buf.type_;
                buffer_ = buf.buffer_;
                buf.size_ = 0;
                buf.capacity_ = 0;
                buf.buffer_ = nullptr;
            }
            return *this;
        }

        void* data(){
            return buffer_;
        }

        const void* data() const{
            return buffer_;
        }

        size_t size() const{
            return size_;
        }

        size_t nbBytes() const{
            return this->size() * getElementSize(type_);
        }

        void resize(size_t newSize){
            size_ = newSize;
            if (capacity_ != newSize){
                free_fn_(buffer_);
                if (!alloc_fn_(&buffer_, this->nbBytes())){
                    throw std::bad_alloc{};
                }
                capacity_ = newSize;
            }
        }

        void resize(const nvinfer1::Dims& dims){
            return this->resize(volume(dims));
        }

        ~GenericBuffer(){
            free_fn_(buffer_);
        }

    private:
        size_t size_{0}, capacity_{0};
        nvinfer1::DataType type_;
        void* buffer_;
        AllocFunc alloc_fn_;
        FreeFunc free_fn_;
};

class DeviceAllocator
{
    public:
        bool operator()(void** ptr, size_t size) const{
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
};

class DeviceFree
{
    public:
        void operator()(void* ptr) const{
            cudaFree(ptr);
        }
};

class HostAllocator
{
    public:
        bool operator()(void** ptr, size_t size) const{
            *ptr = malloc(size);
            return *ptr != nullptr;
        }
};

class HostFree
{
    public:
        void operator()(void* ptr) const{
            free(ptr);
        }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer
{
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
};

}
