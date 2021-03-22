/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <vector>
#include <torch/script.h>

namespace gemfield_org{
    class PriorBox{
        public:
            PriorBox(std::vector<std::vector<int>>&& min_sizes, std::vector<int>&& steps, bool clip = false):
                min_sizes_(min_sizes),steps_(steps),clip_(clip){}
            ~PriorBox() = default;
            at::Tensor forward(std::vector<int>& img_size);
            at::Tensor forward(std::vector<int>&& img_size);

        private:
            std::vector<std::vector<int>> min_sizes_;
            std::vector<int> steps_;
            bool clip_;
    };
}
