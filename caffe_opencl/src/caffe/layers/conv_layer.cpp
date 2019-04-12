#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template<typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
        + 1;
    const int_tp output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
	  /*
	  //std::cout << "bias_term: " << this->bias_term_ << std::endl;
	  {
		  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;

		  int N = top[0]->shape(0);
		  int C = top[0]->shape(1);
		  int H = top[0]->shape(2);
		  int W = top[0]->shape(3);

		  const Dtype *input = top[0]->cpu_data();
		  for (int n = 0; n < N; n++) {
			  for (int c = 0; c < 1; c++) {
				  for (int h = 0; h < 3; h++) {
					  for (int w = 0; w < W; w++) {
						  int out_idx = (((n * C + c) * H) + h) * W + w;
						  std::cout << input[out_idx] << " ";
					  }
					  std::cout << std::endl;
				  }
			  }
		  }
	  }
	  */
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  /*
  if (strcmp("conv1_1", this->layer_param_.name().c_str()) == 0) {
	  std::cout << "############################################################" << std::endl;
	  std::cout << "############################################################" << std::endl;
	  std::cout << "############################################################" << std::endl;

	  int N_b = bottom[0]->shape(0);
	  int C_b = bottom[0]->shape(1);
	  int H_b = bottom[0]->shape(2);
	  int W_b = bottom[0]->shape(3);

	  const Dtype *input_b = bottom[0]->cpu_data();
	  for (int n = 0; n < N_b; n++) {
		  for (int c = 0; c < 1; c++) {
			  for (int h = 0; h < 3; h++) {
				  for (int w = 0; w < W_b; w++) {
					  int out_idx = (((n * C_b + c) * H_b) + h) * W_b + w;
					  std::cout << input_b[out_idx] << " ";
				  }
				  std::cout << std::endl;
			  }
		  }
	  }
	  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;

	  int N = top[0]->shape(0);
	  int C = top[0]->shape(1);
	  int H = top[0]->shape(2);
	  int W = top[0]->shape(3);

	  const Dtype *input = top[0]->cpu_data();
	  for (int n = 0; n < N; n++) {
		  for (int c = 0; c < 1; c++) {
			  for (int h = 0; h < 3; h++) {
				  for (int w = 0; w < W; w++) {
					  int out_idx = (((n * C + c) * H) + h) * W + w;
					  std::cout << input[out_idx] << " ";
				  }
				  std::cout << std::endl;
			  }
		  }
	  }
  }
  */
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
