#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
  /*
  // if (strcmp("conv6_2_CPM", this->layer_param_.name().c_str()) == 0) {
  std::cout << "############################################################" << std::endl;
  std::cout << "############################################################" << std::endl;
  std::cout << "############################################################" << std::endl;

  int N = top[0]->shape(0);
  int C = top[0]->shape(1);
  int H = top[0]->shape(2);
  int W = top[0]->shape(3);

  const Dtype *input = top[0]->cpu_data();
  const Dtype *input2 = bottom[0]->cpu_data();
  for (int n = 0; n < N; n++) {
	  for (int c = 0; c < C; c++) {
		  for (int h = 0; h < H; h++) {
			  for (int w = 0; w < W; w++) {
				  int out_idx = (((n * C + c) * H) + h) * W + w;
				  std::cout << input[out_idx] << " ";
			  }
			  std::cout << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			  for (int w = 0; w < W; w++) {
				  int out_idx = (((n * C + c) * H) + h) * W + w;
				  std::cout << input2[out_idx] << " ";
			  }
			  std::cout << std::endl << "%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
			  std::cout << std::endl;
		  }
	  }
  }
  // }
  */
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int_tp i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
