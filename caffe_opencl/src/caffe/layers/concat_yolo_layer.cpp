#include <vector>

#include "caffe/layers/concat_yolo_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/yolo_v3_loss_layer.hpp"

 
namespace caffe {

template <typename Dtype>
void ConcatYoloLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConcatYoloParameter concat_yolo_param = this->layer_param_.concat_yolo_param();
  nms_thresh = concat_yolo_param.nms_thresh();
}

template <typename Dtype>
void ConcatYoloLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ConcatYoloLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), Dtype(0), top_data);
  
  vector<vector<Detection<Dtype > > > dets;
  int batch_number = bottom[0]->shape(0);
  int class_number = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  for(int b = 0; b < batch_number; ++b){
	vector<Detection<Dtype > > det;
	for(int c = 0; c < class_number; ++c){
  		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			for(int h = 0; h < height; ++h){
				int index = b * class_number * height * width + c * height * width + h * width;	
				Detection<Dtype> single_det;
				single_det.prob = vector<float>(class_number, 0.0);
				single_det.bbox.x = bottom_data[index + 0];
				single_det.bbox.y = bottom_data[index + 1];
				single_det.bbox.w = bottom_data[index + 2];
				single_det.bbox.h = bottom_data[index + 3];
				single_det.objectness = bottom_data[index + 4];
				single_det.prob[c] = bottom_data[index + 5];
				det.push_back(single_det);
			}
		}
	}
	dets.push_back(det);
  }

  for(int i = 0; i < dets.size(); ++i){
	nmsSort(dets[i], class_number, nms_thresh);
  }

  Dtype *p_result_detection = top[0]->mutable_cpu_data();
  int dect_param_number = top[0]->shape(3);
  int top_k = height;
  vector<int> class_index(class_number, 0);
  // caculate each image detail
  for(int b = 0; b < dets.size(); ++b){
	vector<Detection<Dtype> > det = dets[b];	
	// caculate each detection information
	for(int d = 0; d < det.size(); ++d){
		// get the max prob class information
		float max_prob = -1;
		int max_class_prob_index = -1;
		for(int n = 0; n < class_number; ++n){
			if(det[d].objectness > 1e-6 && det[d].prob[n] > 1e-6 && det[d].prob[n] > max_prob){
				max_prob = det[d].prob[n];
				max_class_prob_index = n;
			}
		}
		if(max_prob > 1e-6){
			// get top k detection rectangle
			if(class_index[max_class_prob_index] > top_k){
				continue;
			}
			int index = b * class_number * top_k * dect_param_number + max_class_prob_index * top_k * dect_param_number + class_index[max_class_prob_index] * dect_param_number;
			class_index[max_class_prob_index] += 1;
			p_result_detection[index + 0] = det[d].bbox.x;
			p_result_detection[index + 1] = det[d].bbox.y;
			p_result_detection[index + 2] = det[d].bbox.w;
			p_result_detection[index + 3] = det[d].bbox.h;
			p_result_detection[index + 4] = det[d].objectness;
			p_result_detection[index + 5] = det[d].prob[max_class_prob_index];
		}
  	}
  }
}

template <typename Dtype>
void ConcatYoloLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
/*
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < num_concats_; ++n) {
        caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
            (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
            bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
    offset_concat_axis += bottom_concat_axis;
  }
*/
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ConcatYoloLayer, Forward);
#endif

INSTANTIATE_CLASS(ConcatYoloLayer);
REGISTER_LAYER_CLASS(ConcatYolo);

}  // namespace caffe
