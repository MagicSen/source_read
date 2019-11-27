#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(resize_channels_forward,Dtype)(const int_tp n,
                                              __global const Dtype* in,
                                              const int C, const int H, 
                                              const int W, const KERNEL_ARG_DTYPE scale_,
                                              __global Dtype* out) {
	const int number_size = C * H * W;
	const int channel_size = H * W;
	const int in_height = H / scale_;
	const int in_width = W / scale_;
	for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
		int n = index / number_size;
		int c = (index / channel_size) % C;
		int h = (index / W) % H;
		int w = index % W;
	  	int nw = w/scale_;
	  	int nh = h/scale_;
		int in_idx = (((n * C + c) * in_height) + nh) * in_width + nw;
		out[index] = in[in_idx];
  	}
}