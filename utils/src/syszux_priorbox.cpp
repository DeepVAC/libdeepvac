/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_priorbox.h"
#include <cmath>

namespace gemfield_org{
at::Tensor PriorBox::forward(std::vector<int>& img_size){
	std::vector<float> anchors;
	std::vector<std::vector<int>> feature_maps;

	for(int i=0; i<steps_.size(); i++){
		feature_maps.push_back({ static_cast<int>(std::ceil(1.0 * img_size[0]/steps_[i])), static_cast<int>(std::ceil(1.0 * img_size[1]/steps_[i])) });
	}
	
	for(int i=0; i<feature_maps.size(); i++){
		std::vector<std::vector<int>> product_vec;
		int fh = feature_maps[i][0];
		int fw = feature_maps[i][1];
		for(int h = 0; h < fh; h++){
			for(int w = 0; w < fw; w++){
				for(auto min_size : min_sizes_[i]){
					float s_kx = 1.0 * min_size / img_size[1];
					float s_ky = 1.0 * min_size / img_size[0];
					float dense_cx = (w + 0.5) * steps_[i] / img_size[1];
					float dense_cy = (h + 0.5) * steps_[i] / img_size[0];
					anchors.insert(anchors.end(), {dense_cx, dense_cy, s_kx, s_ky});
				}
			}
		}
	}

	at::Tensor output = at::tensor(anchors).view({-1,4});

	if(clip_){
		output = torch::clamp(output, 1, 0);
	}
	return output;
}
}//namespace