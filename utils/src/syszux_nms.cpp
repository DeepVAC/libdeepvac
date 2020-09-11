/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_nms.h"

namespace{
torch::Tensor nms(torch::Tensor& dets, float threshold)
{
    //1 represent column
	torch::Tensor all_box_x1 = dets.select(1, 0);
	torch::Tensor all_box_y1 = dets.select(1, 1);
	torch::Tensor all_box_x2 = dets.select(1, 2);
	torch::Tensor all_box_y2 = dets.select(1, 3);
	torch::Tensor all_box_scores = dets.select(1, 4);

	int keep_index = 0;
	torch::Tensor keep = torch::zeros({all_box_scores.size(0)}).to(torch::kLong).to(all_box_scores.device());

	torch::Tensor all_box_areas = (all_box_x2 - all_box_x1 + 1) * (all_box_y2 - all_box_y1 + 1);
	std::tuple<torch::Tensor,torch::Tensor> sorted_score_and_index = torch::sort(all_box_scores, 0, 1);
	torch::Tensor sorted_index = std::get<1>(sorted_score_and_index);
	
	//torch::Tensor second2last_box_x1, second2last_box_y1, second2last_box_x2, second2last_box_y2, second2last_box_w, second2last_box_h;

	while(sorted_index.numel() > 0){
		auto top_score_index = sorted_index[0];
		keep[keep_index] = top_score_index;
		keep_index += 1;
		if(sorted_index.size(0)==1){
			break;
        }

		auto second2last_index = sorted_index.slice(0, 1, sorted_index.size(0));
		auto second2last_box_x1 = all_box_x1.index_select(0, second2last_index);
		auto second2last_box_y1 = all_box_y1.index_select(0, second2last_index);
		auto second2last_box_x2 = all_box_x2.index_select(0, second2last_index);
		auto second2last_box_y2 = all_box_y2.index_select(0, second2last_index);

		auto second2last_intersection_x1 = second2last_box_x1.clamp(all_box_x1[top_score_index].item().toFloat(), INT_MAX*1.0);
		auto second2last_intersection_y1 = second2last_box_y1.clamp(all_box_y1[top_score_index].item().toFloat(), INT_MAX*1.0);
		auto second2last_intersection_x2 = second2last_box_x2.clamp(INT_MIN*1.0, all_box_x2[top_score_index].item().toFloat());
		auto second2last_intersection_y2 = second2last_box_y2.clamp(INT_MIN*1.0, all_box_y2[top_score_index].item().toFloat());

		auto second2last_intersection_w = second2last_intersection_x2 - second2last_intersection_x1 + 1;
		auto second2last_intersection_h = second2last_intersection_y2 - second2last_intersection_y1 + 1;
        //change negative w,h to 0
		second2last_intersection_w = second2last_intersection_w.clamp(0., INT_MAX*1.0);
		second2last_intersection_h = second2last_intersection_h.clamp(0., INT_MAX*1.0);

		torch::Tensor intersection_areas_between_top_and_others = second2last_intersection_w * second2last_intersection_h;

		torch::Tensor second2last_box_areas = all_box_areas.index_select(0, second2last_index);
		
		torch::Tensor union_areas_between_top_and_others = (second2last_box_areas - intersection_areas_between_top_and_others) + all_box_areas[top_score_index];
		torch::Tensor iou_between_top_and_others = intersection_areas_between_top_and_others * 1.0 / union_areas_between_top_and_others;
		torch::Tensor index_whether_swallowed_by_top = iou_between_top_and_others <= threshold;
		auto sorted_index_should_left = torch::nonzero(index_whether_swallowed_by_top).squeeze();
		sorted_index = second2last_index.index_select(0, sorted_index_should_left);
	}
	return keep;
}
}//nms