/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_ocr_db.h"
#include "gemfield.h"
#include <assert.h>

using namespace deepvac;

inline std::string BoolToString(bool b)
{
    return b ? "true" : "false";
}

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

    std::vector<std::vector<bool>> perm;
    std::vector<bool> tf {false, true};
    for(int i=0; i<tf.size(); ++i){
        for(int j=0; j<tf.size(); ++j){
            for(int k=0; k<tf.size(); ++k){
                for(int v=0; v<tf.size(); ++v){
                    std::vector<bool> item {tf[i], tf[j], tf[k], tf[v]};
                    perm.push_back(item);
                }
            }
        }
    }
    auto mat_opt = gemfield_org::img2CvMat(path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }

    for(int idx=0; idx<perm.size(); ++idx){
        ocr_detect.setGlab(perm[idx][0]);
        ocr_detect.setExtend(perm[idx][1]);
        ocr_detect.setUnclip(perm[idx][2]);
        ocr_detect.setPolygonScore(perm[idx][3]);
        std::string save_name = "glab_"+BoolToString(perm[idx][0])+"_extend_"+BoolToString(perm[idx][1])+"_unclip_"+BoolToString(perm[idx][2])+"_polygonscore_"+BoolToString(perm[idx][3])+".jpg";
        cv::Mat mat_out;
        mat_opt.value().copyTo(mat_out);
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
        cv::Mat mat_vis;
        mat_opt.value().copyTo(mat_vis);
        for (int i=0; i<rects.size(); ++i){
            std::vector<cv::Point> vPolygonPoint;
            int pts = rects[i].size();
            assert(pts%2==0);
            for(int j=0; j<pts; j=j+2){
                vPolygonPoint.push_back(cv::Point(rects[i][j],rects[i][j+1]));
            }
	        cv::polylines(mat_vis, vPolygonPoint, true, cv::Scalar(0, 0, 255), 2, 4);
        }
        cv::imwrite(save_name, mat_vis);
    }
    return 0;
}
