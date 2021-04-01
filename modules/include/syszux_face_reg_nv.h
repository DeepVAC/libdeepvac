#pragma once

#include <tuple>
#include <vector>
#include "deepvac_nv.h"
#include "syszux_tensorrt_buffers.h"
#include "syszux_img2tensor.h"

namespace deepvac{
class SyszuxFaceRegNV : public DeepvacNV{
    public:
        SyszuxFaceRegNV(std::string path, std::string device = "cpu");
        SyszuxFaceRegNV(std::vector<unsigned char>&& buffer, std::string device = "cpu");
        SyszuxFaceRegNV(const SyszuxFaceRegNV&) = delete;
        SyszuxFaceRegNV& operator=(const SyszuxFaceRegNV&) = delete;
        SyszuxFaceRegNV(SyszuxFaceRegNV&&) = default;
        SyszuxFaceRegNV& operator=(SyszuxFaceRegNV&&) = default;
        virtual ~SyszuxFaceRegNV() = default;
        virtual std::tuple<int, std::string, float> process(cv::Mat& frame);
        void initBinding();
};
}//namespace


