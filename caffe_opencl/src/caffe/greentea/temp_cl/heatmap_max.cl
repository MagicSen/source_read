#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(heatmap_max,Dtype)(const int_tp nthread,
                                              __global const Dtype* in,
                                              const int channels, const int rows, 
                                              const int cols, const KERNEL_ARG_DTYPE scale_,
                                              __global Dtype* out) {
	float ratio = 1 / scale_;
	int center_x = cols / 2 + 1;
	int center_y = rows / 2 + 1;
	int resize_center_x = cols * ratio / 2 + 1;
	int resize_center_y = rows * ratio / 2 + 1;
	float scale_x = scale_;
	float scale_y = scale_;
	for (int_tp index = get_global_id(0); index < nthread; index += get_global_size(0)) 
	{
		int n = index / channels;
		int c = index % channels;
		__global const Dtype* bottom_slice = in
	        + (n * channels + c) * rows * cols;
        Dtype maxval = -DTYPE_MAX;
    	int max_h = -1;	
    	int max_w = -1;	
	    for (int_tp h = 0; h < rows; ++h) 
	    {
	      for (int_tp w = 0; w < cols; ++w) 
	      {
	        if (bottom_slice[h * cols + w] > maxval) 
	        {
	          max_h = h;
	          max_w = w;
	          maxval = bottom_slice[h * cols + w];
	        }
	      }
	    }

	    int small_center_x = (max_w - center_x) * ratio + resize_center_x;
		int small_center_y = (max_h - center_y) * ratio + resize_center_y;

	    for (int mm = -2 * ratio; mm <= 2 * ratio; ++mm)
	    {
			for (int nn = -2 * ratio; nn <= 2 * ratio; ++nn)
			{
				int i = small_center_x + mm
				int j = small_center_y + nn;
				i = i < 0 ? 0 : (i > cols * ratio ? cols * ratio - 1 : i);
				j = j < 0 ? 0 : (j > rows * ratio ? rows * ratio - 1 : j);

	    		float fy = (float)((j + 0.5) * scale_y - 0.5);
				int sy = floor(fy);
				fy -= sy;
				sy = min(sy, rows - 3);
				sy = max(1, sy);

				const float A = -0.75f;

				float coeffsY[4];
				coeffsY[0] = ((A*(fy + 1) - 5 * A)*(fy + 1) + 8 * A)*(fy + 1) - 4 * A;
				coeffsY[1] = ((A + 2)*fy - (A + 3))*fy*fy + 1;
				coeffsY[2] = ((A + 2)*(1 - fy) - (A + 3))*(1 - fy)*(1 - fy) + 1;
				coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

				float fx = (float)((i + 0.5) * scale_x - 0.5);
				int sx = floor(fx);
				fx -= sx;

				if (sx < 1) {
					fx = 0, sx = 1;
				}
				if (sx >= cols - 3) {
					fx = 0, sx = cols - 3;
				}

				float coeffsX[4];
				coeffsX[0] = ((A*(fx + 1) - 5 * A)*(fx + 1) + 8 * A)*(fx + 1) - 4 * A;
				coeffsX[1] = ((A + 2)*fx - (A + 3))*fx*fx + 1;
				coeffsX[2] = ((A + 2)*(1 - fx) - (A + 3))*(1 - fx)*(1 - fx) + 1;
				coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

				float value_all = 0;
				for (int bias_x = -1; bias_x <= 2; ++bias_x){
					for (int bias_y = -1; bias_y <= 2; ++bias_y){
						value_all = value_all + bottom_slice[(sy + bias_y) * cols + sx + bias_x] * coeffsX[bias_x + 1] * coeffsY[bias_y + 1];
					}
				}
				value_all = abs(value_all);
				if (maxval < value){
					maxval = value;
					max_w = new_x;
					max_h = new_y;
				}
		    }
  		}
		top_data[index * 3] = maxval;
		top_data[index * 3 + 1] = max_w;
		top_data[index * 3 + 2] = max_h;	
  	}
}