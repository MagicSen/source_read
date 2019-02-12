#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
// This is changed from accuracy layer
template <typename Dtype>
void LabelMixLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 得到相邻节点聚类的个数
  Mix_Num_ = this->layer_param_.label_mix_param().mix_num();
}

template <typename Dtype>
void LabelMixLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 得到样本的map数量以及channel数目(channel数目等价于joint的数目)
  num_ = bottom[0]->num();
  channel_num_ = bottom[0]->channels();
  inner_num_ = (bottom[0]->height()*bottom[0]->width());

  // mix数组必须等于joints的数目
  CHECK_EQ(channel_num_ , bottom[1]->channels())
      << "Number of labels channels must match number of mix; ";
  // 顶部joint交叉融合，构成一个矩阵，其中每个channel层又分为13个子聚类channel
  top[0]->Reshape(num_, (channel_num_*Mix_Num_), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void LabelMixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* mixtype = bottom[1]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();


  for(int n = 0; n < num_; ++n)
  {
    for(int i = 0; i< channel_num_; ++i)
    {
      // 混合的keypoint index 为 0*13, 1*13, 2*13, 3*13 ... 26*13
      // 最初的相邻关节点都是13类没有编号区分，这里增加偏移，来区分不同关节点的聚类结果
      mixtype[n*channel_num_+i] = mixtype[n*channel_num_+i]+13*i;
    }
  }


  int output_channel = top[0]->channels();
 // LOG(INFO) << "output_channel: " <<output_channel;
  int output_num = top[0]->count();
  for(int i=0; i<output_num; ++i){top_data[i]=0;}

  // 1对13倍
  int bottom_idx = 0, top_idx = 0;
  for (int n = 0; n < num_; ++n) 
  {
    for (int i = 0; i < channel_num_; ++i) 
    {
      int this_mix = mixtype[n*channel_num_+i];
    //  LOG(INFO) << "n: " << n << " this_mix" << this_mix;
      for(int j = 0; j < inner_num_ ; ++j)
      {
        // 图像map的下标
        bottom_idx = n*channel_num_*inner_num_ + i*inner_num_ + j;
        // 顶层混合map的下标
        // 将同一关节点，相邻结点聚类结果，这里类别是1~13，为了取下标要减1
        top_idx    = n*output_channel*inner_num_ + (this_mix-1)*inner_num_ + j;
        top_data[top_idx] = bottom_data[bottom_idx];
      }
    }
  }
}

INSTANTIATE_CLASS(LabelMixLayer);
REGISTER_LAYER_CLASS(LabelMix);

}  // namespace caffe
