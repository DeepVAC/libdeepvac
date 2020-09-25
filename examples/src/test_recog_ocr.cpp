/*
* Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
* This file is part of libdeepvac, licensed under the GPLv3 (the "License")
* You may not use this file except in compliance with the License.
*/

#include "syszux_ocr_recognition.h"
#include "gemfield.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        GEMFIELD_E("usage: deepvac <device> <img_path>");
        return -1;
    }

    std::string device = argv[1];
    std::string path = argv[2];

    cv::Mat img_raw = cv::imread(path);
    if(img_raw.data == nullptr) {
        std::cerr<< path << " is not image file!" << std::endl;
    }

    SyszuxOcrRecognition ocr_recognition(device);
    ocr_recognition(img_raw);
    //strLabelConverter converter("/home/lihang/CRNN_Chinese_Characters_Finetuning/lib/config/chars.txt");
    return 0;
}
