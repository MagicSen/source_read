#ifndef CAFFE_YOLO_V3_LOSS_LAYER_HPP_
#define CAFFE_YOLO_V3_LOSS_LAYER_HPP_

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <vector>
#include <fstream> 
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace boost::property_tree;  // NOLINT(build/namespaces)

namespace caffe {

template <typename Dtype>
struct Box{
  Dtype x;
  Dtype y;
  Dtype w;
  Dtype h;
  Dtype label;
  Box(){ x = -1; y = -1; w = -1; h = -1; label = -1;}
  bool isAvailable(){
	if(x < 1e-8){
		return false;
	}else{
		return true;
	}
  }
};



template <typename Dtype>
struct Detection{
  Box<Dtype> bbox;
  int classes;
  vector<float> prob;
  float objectness;
  int sort_class;
  Detection() { bbox = Box<Dtype>(); classes = -1; prob = vector<float>(0, 0); objectness = 0; sort_class = -1;}
};

#ifdef USE_OPENCV
template <typename Dtype>
cv::Mat drawGroundTruth(cv::Mat img, vector<Box<Dtype> > ground_truth_box);

template <typename Dtype>
cv::Mat drawDetectionResult(cv::Mat img, vector<Detection<Dtype> > single_det, float thresh);

template <typename Dtype>
bool cmp(Detection<Dtype> a, Detection<Dtype> b);

template <typename Dtype>
void nmsSort(vector<Detection<Dtype > > &dets, int class_number, float thresh);

template <typename Dtype>
cv::Mat getOpenCVImage(const Dtype *input_data, const int n_width, const int n_height, const int n_channels);

#endif

int getIndex(const int batch_number, const int location, int skip_channels, int w, int h, int layer_output_number, int class_number);

template <typename Dtype>
Box<Dtype> getYoloBox(const Dtype *x, const vector<vector<float> > anchor_boxes, const int anchor_index, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, const int stride);

template <typename Dtype>
Box<Dtype> getGroundTruth(const Dtype *y, const int index, const int stride);

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);

template <typename Dtype>
Dtype boxIntersection(Box<Dtype> a, Box<Dtype> b);

template <typename Dtype>
Dtype boxUnion(Box<Dtype> a, Box<Dtype> b);

template <typename Dtype>
Dtype boxIOU(Box<Dtype> a, Box<Dtype> b);

template <typename Dtype>
void deltaYoloClass(const Dtype* x, Dtype *y, const int class_index, const int class_idx, const int class_number, const int stride, float *avg_cat);

template <typename Dtype>
Dtype deltaYoloBox(Box<Dtype> truth, const Dtype *x, vector<vector<float> > anchor_boxes, int anchor_index, int index, int i, int j, int lw, int lh, int w, int h, Dtype *y, float scale, int stride);

int getPosInAnchorIndex(vector<int> anchor_index, int index);

template <typename Dtype>
class YoloV3LossLayer : public Layer<Dtype> {
 public:
  explicit YoloV3LossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), diff_(), temp_top(){}

  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "YoloV3Loss"; }

  // only two input: one for predict labels, the other for ground truth.
  // Load src img for test
  //virtual inline int ExactNumBottomBlobs() const { return 3; }
  //virtual int ExactNumBottomBlobs() const;

  // automatically allocate a single top Blob for losslayer.
  virtual inline bool AutoTopBlobs() const { return true; }

  // set output only 1.
  virtual inline int ExactNumTopBlobs() const { return 1; }

  // we cannot backpropagate to labels; ignore force_backward for these inputs.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // base loss layer.
    //return bottom_index != 1;
    // euclidean loss
    return true;
  }
  // must use after forward
  // for test phase
  vector<vector<Detection<Dtype> > > getYoloDetection(float thresh);


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  NOT_IMPLEMENTED;
  }
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // save residual
  Blob<Dtype> diff_;
  Blob<Dtype> temp_top;
  std::string yolo_name;
  int max_box_number;
  float ignore_thresh;
  float truth_thresh;
  int class_number;
  int input_width;
  int input_height;
  vector<int> anchor_index;
  vector<vector<float> > anchor_boxes;
  bool is_debug;
  bool is_show_more_information;
  bool is_train;
  float box_thresh;
  float nms_thresh;
  int top_k; 
};

}  // namespace caffe

#endif  // CAFFE_YOLO_V3_LOSS_LAYER_HPP_
