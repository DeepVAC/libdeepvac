/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_adaptor.h"

namespace gemfield_org {

void get_kernals(torch::Tensor input_data, std::vector<cv::Mat> &kernals) {
    for (int i = 0; i < input_data.size(0); ++i) {
        cv::Mat kernal(input_data[i].size(0), input_data[i].size(1), CV_8UC1);
        std::memcpy((void *) kernal.data, input_data[i].data_ptr(), sizeof(torch::kU8) * input_data[i].numel());
        kernals.emplace_back(kernal);
    }
}

void growing_text_line(std::vector<cv::Mat> &kernals, std::vector<std::vector<int>> &text_line, float min_area) {
    cv::Mat label_mat;
    int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);

    int area[label_num + 1];
    memset(area, 0, sizeof(area));
    for (int x = 0; x < label_mat.rows; ++x) {
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            area[label] += 1;
        }
    }

    std::queue<cv::Point> queue, next_queue;
    for (int x = 0; x < label_mat.rows; ++x) {
        std::vector<int> row(label_mat.cols);
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
                
            if (label == 0) continue;
            if (area[label] < min_area) continue;
                
            cv::Point point(x, y);
            queue.push(point);
            row[y] = label;
        }
        text_line.emplace_back(row);
    }
        
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
        while (!queue.empty()) {
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int label = text_line[x][y];

            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int tmp_x = x + dx[d];
                int tmp_y = y + dy[d];

                if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                if (text_line[tmp_x][tmp_y] > 0) continue;

                cv::Point point(tmp_x, tmp_y);
                queue.push(point);
                text_line[tmp_x][tmp_y] = label;
                is_edge = false;
            }

            if (is_edge) {
                next_queue.push(point);
            }
        }
        swap(queue, next_queue);
    }
} 

std::vector<std::vector<int>> adaptor_pse(torch::Tensor input_data, float min_area) {
    std::vector<cv::Mat> kernals;
    get_kernals(input_data, kernals);

    std::vector<std::vector<int>> text_line;
    growing_text_line(kernals, text_line, min_area);
    return text_line;
}
}//namespace

