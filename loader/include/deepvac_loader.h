/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <tuple>
#include <filesystem>
#include <unordered_set>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
namespace deepvac{

std::optional<cv::Mat> img2CvMat(std::string& img_path, bool is_rgb=false);
std::optional<at::Tensor> img2tensor(std::string& img_path, bool is_rgb=false);

template <typename SyszuxWalkerType>
class DeepvacIterBase {
    public:
        DeepvacIterBase(){}
        DeepvacIterBase(SyszuxWalkerType walker): walker_(walker){}
        DeepvacIterBase& operator=(const DeepvacIterBase&) = delete;
        virtual ~DeepvacIterBase() = default;
        DeepvacIterBase operator++() { ++walker_;  return *this; }
        bool operator!=(const DeepvacIterBase & other) const {return walker_ != other.walker_; }

    protected:
        SyszuxWalkerType walker_;
};

using deepvac_sfrdi = std::filesystem::recursive_directory_iterator;
class RecursiveFileIterBase : public DeepvacIterBase<deepvac_sfrdi>{
    public:
        RecursiveFileIterBase(const char* path) : DeepvacIterBase<deepvac_sfrdi>(deepvac_sfrdi(path)) {}
        RecursiveFileIterBase() : DeepvacIterBase<deepvac_sfrdi>(deepvac_sfrdi()) {}
};

class RecursiveFileIter : public RecursiveFileIterBase{
    public:
        RecursiveFileIter(const char* path) : RecursiveFileIterBase(path) {}
        RecursiveFileIter() : RecursiveFileIterBase() {}
        virtual const std::string operator*() const;
};

class CvMatIter : public RecursiveFileIterBase{
    public:
        CvMatIter(const char* path, bool is_rgb=false) : RecursiveFileIterBase(path),is_rgb_(is_rgb) {}
        CvMatIter(bool is_rgb=false) : RecursiveFileIterBase(),is_rgb_(is_rgb) {}
        virtual const std::optional<cv::Mat> operator*() const;
    private:
        bool is_rgb_;
};

class ImgFileInputTensorPairIter : public RecursiveFileIterBase{
    public:
        ImgFileInputTensorPairIter(const char* path, bool is_rgb=false) : RecursiveFileIterBase(path),is_rgb_(is_rgb) {}
        ImgFileInputTensorPairIter(bool is_rgb=false) : RecursiveFileIterBase(),is_rgb_(is_rgb) {}
        virtual const std::tuple<std::string, std::optional<at::Tensor>> operator*() const;
    private:
        bool is_rgb_;
        std::unordered_set<std::string> suffix_ {".jpg",".jpeg",".png"};
};

template <typename DeepvacIterType>
class DeepvacRecursiveFileLoader {
    public:
        DeepvacRecursiveFileLoader(const char* url): url_(url) {}
        DeepvacIterType begin()  { return DeepvacIterType(url_.c_str()); }
        DeepvacIterType end()  { return DeepvacIterType(); }

    protected:
        std::string url_;
};


//loader that based on std container.
using deepvac_sv = std::vector<std::string>;
class DeepvacVectorIter : public DeepvacIterBase<deepvac_sv::iterator>{
    public:
        DeepvacVectorIter(deepvac_sv::iterator sv): DeepvacIterBase<deepvac_sv::iterator>(sv) {}
        virtual std::optional<cv::Mat> operator*() const;
};

template <typename DeepvacIterType>
class DeepvacContainerLoader {
    public:
        DeepvacContainerLoader(deepvac_sv sv) : sv_(std::move(sv)) {}
        DeepvacIterType begin()  { return DeepvacIterType(sv_.begin()); }
        DeepvacIterType end()  { return DeepvacIterType(sv_.end()); }
    private:
        deepvac_sv sv_;
};

}//namespace deepvac