/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include "syszux_cls_resnet.h"
#include "syszux_img2tensor.h"
#include "gemfield.h"
#include "deepvac.h"
#include <chrono>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include "syszux_imagenet_classes.h"

using namespace deepvac;

void validate(SyszuxClsResnet& civilnet, at::Tensor& t, std::string img_path){
    std::cout<<"---------------VALIDATE BEGIN---------------"<<std::endl;
    civilnet.set({static_cast<int>(t.size(4)), static_cast<int>(t.size(3))});
    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected in validate!");
    }
    auto mat_out = mat_opt.value();

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto start = std::chrono::system_clock::now();

    auto resnet_out_opt = civilnet.process(mat_out);

    stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string msg = gemfield_org::format("Overall process time in validate: %f", model_loading_duration.count());

    if(!resnet_out_opt){
        throw std::runtime_error("return empty error!");
    }
    
    auto resnet_out = resnet_out_opt.value();
    std::cout <<msg<< "|Index: " << resnet_out.first << "|Class: " << gemfield_org::imagenet_classes[resnet_out.first] << "|Probability: " << resnet_out.second << std::endl;
    std::cout<<"---------------VALIDATE END---------------"<<std::endl;
}

void warmUp(SyszuxClsResnet& civilnet, at::Tensor& t, std::string img_path, std::string device){
    std::cout<<"---------------WARMUP BEGIN---------------"<<std::endl;
    //for loop to warmup
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(device);

    for(int i=0; i<10;i++){
        auto ti = torch::rand({static_cast<int>(t.size(1)),static_cast<int>(t.size(2)),static_cast<int>(t.size(3)), static_cast<int>(t.size(4))},options);
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto start = std::chrono::system_clock::now();
        auto resnet_out_opt = civilnet.process(ti);
        stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
        std::string msg = gemfield_org::format("Overall process time in warmup: %f", model_loading_duration.count());
    }
    std::cout<<"---------------WARMUP END---------------"<<std::endl;
}

void benchmark(SyszuxClsResnet& civilnet, at::Tensor& t, std::string img_path, std::string device){
    validate(civilnet, t, img_path);
    warmUp(civilnet, t, img_path, device);

    auto item_num = t.size(0);
    std::cout<<"---------------BENCHMARK BEGIN---------------"<<item_num<<std::endl;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto start = std::chrono::system_clock::now();

    for(int i=0; i<item_num; i++){
        auto ti = t[i];
        auto resnet_out_opt = civilnet.process(ti);
    }

    stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::duration<double> model_loading_duration = std::chrono::system_clock::now() - start;
    std::string header = "|Model|Engine|Input size|forward time|\n";
    std::string header2 = "|-----|-------|----------|-----------|\n";
    std::string msg = gemfield_org::format("|Resnet50|libtorch|%dx%d|%f|\n", t.size(4),t.size(3),model_loading_duration.count()/item_num);
    std::cout << header<<header2<<msg<< std::endl;
    std::cout<<"---------------BENCHMARK END---------------"<<std::endl;
}
int main(int argc, const char* argv[]) {
    if (argc != 4) {
        GEMFIELD_E("usage: test_resnet_benchmark <device> <model_path> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string model_path = argv[2];
    std::string img_path = argv[3];

    std::cout << "device : " << device << std::endl;
    std::cout << "model_path : " << model_path << std::endl;
    std::cout << "img_path : " << img_path << std::endl;

    //at::init_num_threads();
    //at::set_num_threads(16);
    //at::set_num_interop_threads(16);
    std::cout<<"userEnabledCuDNN: "<<at::globalContext().userEnabledCuDNN()<<std::endl;
    std::cout<<"userEnabledMkldnn: "<<at::globalContext().userEnabledMkldnn()<<std::endl;
    std::cout<<"benchmarkCuDNN: "<<at::globalContext().benchmarkCuDNN()<<std::endl;
    std::cout<<"deterministicCuDNN: "<<at::globalContext().deterministicCuDNN()<<std::endl;
    std::cout<<"gemfield thread num: "<<at::get_num_threads()<<" | "<<at::get_num_interop_threads()<<std::endl;

    SyszuxClsResnet cls_resnet(model_path, device);
    //step1. init some random tensors as input, need 4GB GPU RAM.
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(device);
    auto t224x224 = torch::rand({100,1,3,224,224}, options);
    auto t640x640 = torch::rand({100,1,3,640,640}, options);
    auto t1280x720 = torch::rand({50,1,3,720,1280}, options);
    auto t1280x1280 = torch::rand({50,1,3,1280,1280}, options);
   
    //step2 benchmark
    benchmark(cls_resnet, t224x224, img_path, device);
    benchmark(cls_resnet, t640x640, img_path, device);
    benchmark(cls_resnet, t1280x720, img_path, device);
    benchmark(cls_resnet, t1280x1280, img_path, device);
    return 0;
}
