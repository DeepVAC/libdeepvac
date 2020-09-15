/*
 *  * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 *   * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 *    * You may not use this file except in compliance with the License.
 *     */

#include "syszux_filter_face.h"

int main(int argc, char** argv){
    if (argc != 2) {
        std::cout << "usage: test_syszux_alignface <img_path>" << std::endl;
        return -1;	
    }	        
    std::string img_path = argv[1];
    cv::Mat img = cv::imread(img_path);
    std::vector<float> bbox = {608.8696,131.5066,726.54895,276.54346};
    std::vector<float> landmark = {632.30804,177.63857,687.6927,185.6925,649.9065,
                                 213.16966,632.7789,239.9633,673.7113,246.61923};
    bool res = gemfield_org::filterFace(img, bbox, landmark);
    std::cout << res << std::endl;
}
