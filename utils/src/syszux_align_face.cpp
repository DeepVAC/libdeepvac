/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include <vector>
#include "gemfield.h"
#include "syszux_align_face.h"

namespace gemfield_org{
AlignFace::AlignFace()
{
    float REFERENCE_FACIAL_POINTS_112x112[5][2] = {
	{38.29459953, 51.69630051},
	{73.53179932, 51.50139999},
	{56.02519989, 71.73660278},
	{41.54930115, 92.3655014},
	{70.72990036, 92.20410156}
    };
    ref_facial_5pts_ = cv::Mat(5, 2, CV_32FC1, &REFERENCE_FACIAL_POINTS_112x112).clone();

    float CROP_SIZE[2] = {112,112};
    crop_size_ = cv::Mat(1,2,CV_32FC1,&CROP_SIZE).clone();
}

cv::Mat AlignFace::operator() (cv::Mat& frame, cv::Mat& facial_5pts)
{
    cv::Mat x, y;
    cv::Mat facial;
    //x1,y1,x2,y2... somebody familiar with cv::Mat, please refactor this code snippet.
    for(int i=0; i<facial_5pts.cols; i=i+2){
        x.push_back(facial_5pts.at<float>(i));
        y.push_back(facial_5pts.at<float>(i+1));
    }
    
    cv::hconcat(x, y, facial);
    return warpAndCrop(frame, facial);
}

cv::Mat AlignFace::warpAndCrop(cv::Mat& src_img, cv::Mat& facial_5pts)
{
    if(ref_facial_5pts_.rows != facial_5pts.rows){
        std::string msg = gemfield_org::format("ref_facial_5pts_ and facial_5pts must have the same shape: %d: vs %d", int(ref_facial_5pts_.rows), int(facial_5pts.rows));
        GEMFIELD_E(msg.c_str());
        throw std::runtime_error(msg);
	}
	
    cv::Mat tfm = getAffineTransform(facial_5pts);

    cv::Mat dst_img;
    cv::warpAffine(src_img, dst_img, tfm, cv::Size(int(crop_size_.at<float>(0)), int(crop_size_.at<float>(1))));

    return dst_img;
}

cv::Mat AlignFace::getAffineTransform(cv::Mat& facial_5pts)
{
    cv::Mat trans1 = findNonereflectiveSimilarity(facial_5pts, ref_facial_5pts_);
	
    cv::Mat xyR = ref_facial_5pts_.clone();
    xyR.col(0) = -1 * xyR.col(0);

    cv::Mat trans2r = findNonereflectiveSimilarity(facial_5pts, xyR);
    
    cv::Mat TreflectY = cv::Mat::eye(3, 3, CV_32F);
    TreflectY.at<float>(0, 0) = -1;
	
    cv::Mat trans2 = trans2r * TreflectY;

    cv::Mat xy1 = tformfwd(trans1, facial_5pts);
    float norm1 = cv::norm(xy1, ref_facial_5pts_);
    cv::Mat xy2 = tformfwd(trans2, facial_5pts);
    float norm2 = cv::norm(xy2, ref_facial_5pts_);

    cv::Mat trans;
    if(norm1 <= norm2){
        trans = trans1;
    }else{
        trans = trans2;
    }
    return trans.colRange(0, 2).t();
}

cv::Mat AlignFace::findNonereflectiveSimilarity(cv::Mat& facial_5pts, cv::Mat& ref_facial_5pts)
{
    int K = 2;
    int M = ref_facial_5pts.rows;
    cv::Mat x = ref_facial_5pts.col(0);
    cv::Mat y = ref_facial_5pts.col(1);

    cv::Mat temp1, temp2, X;
    std::vector<cv::Mat> matrices1 = {x, y, cv::Mat::ones(M, 1, CV_32F), cv::Mat::zeros(M, 1, CV_32F)};
    std::vector<cv::Mat> matrices2 = {y, -x, cv::Mat::zeros(M, 1, CV_32F), cv::Mat::ones(M, 1, CV_32F)};
    
    cv::hconcat(matrices1, temp1);
    cv::hconcat(matrices2, temp2);
    cv::vconcat(temp1, temp2, X);
    
    cv::Mat U;
    cv::Mat u = facial_5pts.col(0);
    cv::Mat v = facial_5pts.col(1);
    cv::vconcat(u, v, U);

    cv::Mat s_r, u_r, v_r;
    cv::SVD::compute(X, s_r, u_r, v_r);
    if(s_r.rows < 2 * K){
        throw std::runtime_error("cp2tform:twoUniquePointsReq");
    }
    cv::Mat r;
    cv::solve(X, U, r, cv::DECOMP_SVD);
    float sc = r.at<float>(0, 0);
    float ss = r.at<float>(1, 0);
    float tx = r.at<float>(2, 0);
    float ty = r.at<float>(3, 0);

    float Tinv_vec[3][3] = {{sc, -ss, 0.},{ss, sc, 0.},{tx, ty, 1.}};
    cv::Mat trans_inv = cv::Mat(3, 3, CV_32F, Tinv_vec);
    cv::Mat trans = trans_inv.inv();

    std::vector<float> tmp_vec = {0, 0, 1};
    cv::Mat tmp(tmp_vec);
    tmp = tmp.t();
    trans.col(2) = tmp;
    return trans;
}

cv::Mat AlignFace::tformfwd(cv::Mat& trans, cv::Mat facial_5pts)
{
    cv::hconcat(facial_5pts, cv::Mat::ones(facial_5pts.rows, 1, CV_32F), facial_5pts);
    cv::Mat res = facial_5pts * trans;
    return res.colRange(0, 2);
}
}//namespace
