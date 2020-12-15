/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

/* 听到项目上传来的好消息，放诗一首：
 * 剑外忽传收蓟北，初闻涕泪满衣裳。
 * 却看妻子愁何在，漫卷诗书喜欲狂。
 * 白日放歌须纵酒，青春作伴好还乡。
 * 即从巴峡穿巫峡，便下襄阳向洛阳。
 */
#pragma once

#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
/*
#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)
 
#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)
 
#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE
 
#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))
 
#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))
 
#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))
 
#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))
 
#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))
 
#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))
 
#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))
 
#define CV_16FC1 CV_MAKETYPE(CV_16F,1)
#define CV_16FC2 CV_MAKETYPE(CV_16F,2)
#define CV_16FC3 CV_MAKETYPE(CV_16F,3)
#define CV_16FC4 CV_MAKETYPE(CV_16F,4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F,(n))
*/

namespace pybind11 {
namespace detail {

//based on https://github.com/pybind/pybind11/issues/538
template <> 
struct type_caster<cv::Mat> {
    //This macro establishes the name 'inty' in function signatures
    //and declares a local variable 'value' of type inty.
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        if (!isinstance<array>(src)){
            throw std::runtime_error("parameter type not array.");
            return false;
        }
        array src_array = reinterpret_borrow<array>(src);
        auto src_array_info = src_array.request();
        //The number of dimensions the memory represents as an n-dimensional array. 
        //If it is 0, buf points to a single item representing a scalar. 
        //In this case, shape, strides and suboffsets MUST be NULL. Gemfield
        int ndims = src_array_info.ndim;
        if(ndims != 3){
            throw std::runtime_error("only support 3d ndarray.");            
            return false;
        }

        decltype(CV_32F) dtype;
        //#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))
        if(src_array_info.format == format_descriptor<float>::format()){
            dtype = CV_32FC(ndims);
        }else if (src_array_info.format == format_descriptor<double>::format()){
            dtype = CV_64FC(ndims);
        }else if (src_array_info.format == format_descriptor<unsigned char>::format()){
            dtype = CV_8UC(ndims); 
        }else if (src_array_info.format == format_descriptor<int>::format()){
            dtype = CV_32SC(ndims);
        }else{
            throw std::runtime_error("Only support float,double,uchar,int.");
            return false;
        }

        //shape[0] * ... * shape[ndim-1] * itemsize MUST be equal to len. Gemfield
        int h = src_array_info.shape[0];
        int w = src_array_info.shape[1];

        value = cv::Mat(h, w, dtype, src_array_info.ptr, cv::Mat::AUTO_STEP);
        return true;
    }
    //convert an inty instance into a Python object. 
    //The second and third arguments are used to indicate the return value policy and parent object
    // (for ``return_value_policy::reference_internal``) and are generally ignored by implicit casters. 
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int mat_w = m.cols;
        int mat_h = m.rows;
        int mat_c = m.channels();
        auto type = m.type();
        auto depth = m.depth();
        int dim = (depth == type)? 2 : 3;

        //enum{CV_8U=0,CV_8S=1,CV_16U=2,CV_16S=3,CV_32S=4,CV_32F=5,CV_64F=6,CV_16F=7}
        switch(depth) {
            case CV_8U:
                format = format_descriptor<unsigned char>::format();
                elemsize = sizeof(unsigned char);
                break;
            case CV_32S:
                format = format_descriptor<int>::format();
                elemsize = sizeof(int);
                break;
            case CV_32F:
                format = format_descriptor<float>::format();
                elemsize = sizeof(float);
                break;
            default: 
                throw std::runtime_error("Unsupported type");
        }
        
        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) mat_h, (size_t) mat_w};
            strides = {elemsize * (size_t) mat_w, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) mat_h, (size_t) mat_w, (size_t) 3};
            strides = {(size_t) elemsize * mat_w * 3, (size_t) elemsize * 3, (size_t) elemsize};
        } else{
            throw std::runtime_error("Unsupported dimension.");
        }
        return array(buffer_info(
            m.data,         /* Pointer to buffer */
            elemsize,       /* Size of one scalar */
            format,         /* Python struct-style format descriptor */
            dim,            /* Number of dimensions */
            bufferdim,      /* Buffer dimensions */
            strides         /* Strides (in bytes) for each index */
            )).release();
    }
};
}//namespace detail
}//namespace pybind11



