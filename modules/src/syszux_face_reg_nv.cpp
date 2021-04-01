#include "syszux_face_reg_nv.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "NvInfer.h"
#include "gemfield.h"

namespace deepvac {
SyszuxFaceRegNV::SyszuxFaceRegNV(std::string path, std::string device):DeepvacNV(path, device) {
    setBinding(4);
    initBinding();
}

SyszuxFaceRegNV::SyszuxFaceRegNV(std::vector<unsigned char>&& buffer, std::string device):DeepvacNV(std::move(buffer), device){
    setBinding(4);
    initBinding();
}

void SyszuxFaceRegNV::initBinding() {
    datas_[0].hostBuffer.resize(nvinfer1::Dims4{1, 3, 112, 112});
    datas_[0].deviceBuffer.resize(nvinfer1::Dims4{1, 3, 112, 112});
    datas_[1].hostBuffer.resize(nvinfer1::Dims2{1, 512});
    datas_[1].deviceBuffer.resize(nvinfer1::Dims2{1, 512});
}

std::tuple<int, std::string, float> SyszuxFaceRegNV::process(cv::Mat& frame) {
    std::optional<std::vector<std::tuple<cv::Mat, std::vector<float>, std::vector<float>>>> faces;
    auto out = faces.value();
    for (int i=0; i<out.size(); i++){
        auto [img, bbox, points] = out[i];
        cv::Mat dst;
        img.convertTo(dst, CV_32F, 1.0 / 127.5, -1.0);
        float* input_data = (float*)dst.data;
        float* hostDataBuffer = static_cast<float*>(datas_[0].hostBuffer.data());
        for (int c = 0; c < 3; ++c) {
            for(int j = 0, volChl=112*112; j < volChl; ++j) {
                hostDataBuffer[c*volChl + j] = input_data[j*3 + c];
            }
        }
        cudaMemcpy(datas_[0].deviceBuffer.data(), datas_[0].hostBuffer.data(), datas_[0].hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
        std::vector<void*> predicitonBindings = {datas_[0].deviceBuffer.data(), datas_[1].deviceBuffer.data()};
        auto predict = forward(predicitonBindings.data());
        cudaMemcpy(datas_[1].hostBuffer.data(), datas_[1].deviceBuffer.data(), datas_[1].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
    }

    int min_index=0;
    float min_distance = 0;
    std::string gemfield = "gemfield";
    return std::make_tuple(min_index, gemfield, min_distance);
}

}//namespace
