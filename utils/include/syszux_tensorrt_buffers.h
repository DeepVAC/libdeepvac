#pragma once

#include <algorithm>
#include <cuda_runtime_api.h>
#include <numeric>
#include <torch/script.h>
#include <vector>

namespace gemfield_org{

enum class DataType : int32_t {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4
};

enum class DeviceType {
    CPU,
    CUDA
};

inline unsigned int getElementSize(DataType t){
    switch (t){
        case DataType::kINT32: return 4;
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kBOOL:
        case DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const std::vector<int64_t>& d){
    return std::accumulate(d.begin(), d.end(), 1, std::multiplies<int64_t>());
}

template <typename AllocFunc, typename FreeFunc, DeviceType Device>
class GenericBuffer
{
    public:
        GenericBuffer(DataType type = DataType::kFLOAT)
            : size_(0), type_(type), buffer_(nullptr), cleanup_(true) {}

        GenericBuffer(void* buffer, const std::vector<int64_t>& shape, const DataType& type): size_(volume(shape)), shape_(shape), type_(type), 
                                                                                            buffer_(buffer), cleanup_(false) {}

        GenericBuffer(GenericBuffer&& buf)
            : size_(buf.size_),  shape_(std::move(buf.shape_)), type_(buf.type_), buffer_(buf.buffer_), cleanup_(buf.cleanup_) {
            buf.reset();
        }

        GenericBuffer& operator=(GenericBuffer&& buf) {
            if (this != &buf) {
                if(cleanup_) {
                    free_fn_(buffer_);
                }
                size_ = buf.size_;
                shape_ = std::move(buf.shape_);
                type_ = buf.type_;
                buffer_ = buf.buffer_;
                cleanup_ = buf.cleanup_;
                buf.reset();
            }
            return *this;
        }

        void* data() {
            return buffer_;
        }

        const void* data() const {
            return buffer_;
        }

        size_t size() const {
            return size_;
        }

        size_t nbBytes() const {
            return this->size() * getElementSize(type_);
        }

        std::vector<int64_t> shape() const {
            return shape_;
        }

        void resize(const std::vector<int64_t>& shape) {
            shape_ = shape;
            auto new_size = volume(shape_);

            if(size_ == new_size) {
                return;
            }
            if(cleanup_) {
                free_fn_(buffer_);
            }
            if(!alloc_fn_(&buffer_, new_size*getElementSize(type_))) {
                throw std::bad_alloc{};
            }
            size_ = new_size;
            cleanup_ = true;
        }

        void fromTensor(const at::Tensor& tensor) {
            auto contiguous_tensor = tensor.contiguous();
            auto* input_data = static_cast<float*>(contiguous_tensor.to("cpu").data_ptr());
            auto sizes = contiguous_tensor.sizes();
            std::vector<int64_t> shape;
            std::copy(sizes.begin(), sizes.end(), std::back_inserter(shape));
            resize(shape);
            if(Device == DeviceType::CPU) {
                memcpy(buffer_, input_data, nbBytes());
            } else if(Device == DeviceType::CUDA) {
                cudaMemcpy(buffer_, input_data, nbBytes(), cudaMemcpyHostToDevice);
            }
        }

        at::Tensor toTensor() {
            return torch::from_blob(buffer_, shape_, Device == DeviceType::CPU ? torch::kCPU : torch::kCUDA);
        }

        ~GenericBuffer() {
            if(cleanup_) {
                free_fn_(buffer_);
            }
        }

    private:
        void reset() {
            size_ = 0;
            buffer_ = nullptr;
            shape_.clear();
        }

    private:
        size_t size_;
        std::vector<int64_t> shape_;
        DataType type_;
        void* buffer_;
        AllocFunc alloc_fn_;
        FreeFunc free_fn_;
        bool cleanup_;
};

class DeviceAllocator
{
    public:
        bool operator()(void** ptr, size_t size) const {
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

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree, DeviceType::CUDA>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree, DeviceType::CPU>;

class ManagedBuffer
{
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
};

}
