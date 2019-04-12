// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <limits>
#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  // 按照threshold_划分，计算uint类型对应的范围
  uint_thres_ =
      static_cast<uint_tp>(static_cast<long double>
          (std::numeric_limits<uint_tp>::max())
          * static_cast<long double>(threshold_));
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 上下层结构输入一致
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  // 生成随机数，放置到缓存里
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  uint_tp* mask = rand_vec_.mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    // 0~1 伯努利分布，生成随机数， 0101011101
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int_tp i = 0; i < count; ++i) {
      // scale保证 数据总量 一致
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    // 测试过程下，直接赋值
    caffe_cpu_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      // rand_vec_ 记录 dropout过程中的舍弃的位置
      const uint_tp* mask = rand_vec_.cpu_data();
      const int_tp count = bottom[0]->count();
      for (int_tp i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      // 测试阶段
      caffe_cpu_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
