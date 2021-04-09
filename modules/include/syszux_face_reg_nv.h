#pragma once

#include <tuple>
#include <torch/torch.h>
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
        virtual std::optional<std::vector<std::tuple<int, std::string, float>>> process(std::vector<cv::Mat>& frames);
        void initBinding();

    private:
        void loadDB(std::string path);
        int size();
        int cachedSize();
        void commit(std::string path_prefix);
        int add(cv::Mat& frame, std::string name);


    private:
        int batch_size_;
        std::vector<at::Tensor> db2commit_vec_;
        std::vector<std::string> id2commit_vec_;
        at::Tensor db_;
        std::vector<std::string> id_vec_;

};
}//namespace


