/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_ocr_db.h"
#include "gemfield.h"
#include <assert.h>

using namespace deepvac;
int main(int argc, char** argv)
{
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }
    std::string device = argv[1];
    std::string path = argv[2];
    int long_size = 1280;
    int crop_gap = 5;
    int text_min_area = 300;
    int text_min_size = 3;
    float text_mean_score = 0.5;
    float text_thresh = 0.3;
    float unclip_ratio=2;

    SyszuxOcrDB ocr_detect;
    ocr_detect.setDevice(device);
    ocr_detect.setModel("/gemfield/hostpv/lihang/github/libdeepvac/install/lib/deepvac/ocr.db.deepvac");

    ocr_detect.set(long_size, crop_gap, text_min_area, text_min_size, text_mean_score, text_thresh, unclip_ratio);
    auto mat_opt = gemfield_org::img2CvMat(path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto mat_out = mat_opt.value();
    auto detect_out_opt = ocr_detect.process(mat_out);
    if(!detect_out_opt){
        throw std::runtime_error("no text detected");
	return 0;
    }

    std::pair<std::vector<cv::Mat>, std::vector<std::vector<int>>> detect_out = detect_out_opt.value();
    std::vector<cv::Mat> crop_imgs = detect_out.first;
    std::vector<std::vector<int>> rects = detect_out.second;

    if (crop_imgs.size()==0) {
        std::cout << "no text detected" << std::endl;
        return 0;
    }
    for (int i=0; i<crop_imgs.size(); i++) {
        cv::imwrite("./ocr_detect_test" + std::to_string(i) + ".jpg", crop_imgs[i]);
        std::cout << "rect: " << rects[i] << std::endl;
    }
    for (int i=0; i<rects.size(); ++i){
        std::vector<cv::Point> vPolygonPoint;
        int pts = rects[i].size();
        assert(pts%2==0);
        for(int j=0; j<pts; j=j+2){
            vPolygonPoint.push_back(cv::Point(rects[i][j],rects[i][j+1]));
        }
	cv::polylines(mat_out, vPolygonPoint, true, cv::Scalar(0, 0, 255), 2, 4);
    }
    cv::imwrite("./vis.jpg", mat_out);
    return 0;
}
