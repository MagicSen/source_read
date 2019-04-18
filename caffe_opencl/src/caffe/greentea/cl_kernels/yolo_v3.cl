#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(yolo_v3_forward,Dtype)(const int_tp n,
                                              __global const Dtype* in,
                                              const int sample_number, const int anchor_size, 
                                              const int grid_size, const int class_number, 
                                              const int grid_w,  const int image_w, 
                                              const int image_h, const KERNEL_ARG_DTYPE box_thresh,
                                              __global Dtype* out) {
  	float anchor_boxes_x[6] = {4.0, 8.0, 15.0, 26.0, 44.0, 106.0};
  	float anchor_boxes_y[6] = {4.0, 9.0, 16.0, 27.0, 46.0, 81.0};
  	int temp_size = anchor_size * grid_size;
  	int each_sample_size = anchor_size * grid_size * (4 + 1 + class_number);
  	int each_anchor_size = grid_size * (4 + 1 + class_number);
  	int grid_h = grid_size / grid_w;
	for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
		int a = index / temp_size;
		int b = (index / grid_size) % anchor_size;
		int c = index % grid_size;
		int j = c / grid_w;
		int i = c % grid_w;
		int start_index = a * each_sample_size + b * each_anchor_size;
		int index_x = start_index + 0 * grid_size;
		int index_y = start_index + 1 * grid_size;
		int index_w = start_index + 2 * grid_size;
		int index_h = start_index + 3 * grid_size;
		int index_objectness = start_index + 4 * grid_size;
		float x = 1.0 / (1.0 + exp(-in[index_x]));
		float y = 1.0 / (1.0 + exp(-in[index_y]));
		float w = in[index_w];
		float h = in[index_h];
		float objectness = 1.0 / (1.0 + exp(-in[index_objectness]));
		x = (i + x) / (grid_w * 1.0);
		y = (j + y) / (grid_h * 1.0);
		w = exp(w) * anchor_boxes_x[b] / image_w;
		h = exp(h) * anchor_boxes_y[b] / image_h;
		out[index_x] = x;
		out[index_y] = y;
		out[index_w] = w;
		out[index_h] = h;
		out[index_objectness] = objectness;
		for(int k = 1; k < class_number+1; ++k){
			float prob_class = 1.0 / (1.0 + exp(-in[index_objectness + k * grid_size]));
			float prob = objectness * prob_class;
			out[index_objectness + k * grid_size] = (prob > box_thresh) ? prob : 0.0;
		}
  	}
}