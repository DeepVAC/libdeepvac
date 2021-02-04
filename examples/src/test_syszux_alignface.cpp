/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "gemfield.h"
#include "syszux_align_face.h"
#include "syszux_img2tensor.h"
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2) {
        GEMFIELD_E("usage: test_syszux_alignface <img_path>");
        return -1;
    }

    std::string img_path = argv[1];
    auto mat_opt = gemfield_org::img2CvMat(img_path);
    if(!mat_opt){
        throw std::runtime_error("illegal image detected");
        return 1;
    }
    auto mat_out = mat_opt.value();

    std::vector<float> facial_5pts = {632.30804, 177.63857, 687.6927, 185.6925, 649.9065,
                                    213.16966, 632.7789, 239.9633, 673.7113, 246.61923};

    cv::Mat facial_5pts_mat(facial_5pts);
    facial_5pts_mat = facial_5pts_mat.t();

    gemfield_org::AlignFace align_face;
    auto [dst_img, dst_points] = align_face(mat_out, facial_5pts_mat);
    std::cout << dst_img.rows << std::endl;
    cv::imwrite("./test_res.jpg", dst_img);
}
