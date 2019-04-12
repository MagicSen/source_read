// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
//#include "caffe/loss_layers.hpp"
#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/gen_anchors.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/bbox_util.hpp"

using namespace boost::property_tree;  // NOLINT(build/namespaces)

namespace caffe {

/* ROIPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class ROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
};

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
   * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  bool has_weights_;
};

/* SimplerNMSLayer - N Mini-batch Sampling Layer
*/
template <typename Dtype>
class SimplerNMSLayer : public Layer<Dtype> {
public:
    SimplerNMSLayer(const LayerParameter& param) :Layer<Dtype>(param),
        max_proposals_(500),
        prob_threshold_(0.5f),
        iou_threshold_(0.7f),
        min_bbox_size_(16),
        feat_stride_(16),
        pre_nms_topN_(6000),
        post_nms_topN_(300) {
    };

    ~SimplerNMSLayer() {
    }

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        top[0]->Reshape(std::vector<int>{ max_proposals_, 5 });
    }

    virtual inline const char* type() const { return "SimplerNMS"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    int max_proposals_;
    float prob_threshold_;
    // TODO: add to proto
    float iou_threshold_;
    int min_bbox_size_;
    int feat_stride_;
    int pre_nms_topN_;
    int post_nms_topN_;

    // relative to center point,
    // currently, it is always float, just do a quick fix
    Blob<float> anchors_blob_;

    //TODO: clamp is part of std as of c++17...
    static inline const Dtype clamp_v(const Dtype v, const Dtype v_min, const Dtype v_max)
    {
        return std::max(v_min, std::min(v, v_max));
    }
    struct simpler_nms_roi_t
    {
        Dtype x0, y0, x1, y1;

        Dtype area() const { return std::max<Dtype>(0, y1 - y0 + 1) * std::max<Dtype>(0, x1 - x0 + 1); }
        simpler_nms_roi_t intersect (simpler_nms_roi_t other) const
        {
            return
            {
                std::max(x0, other.x0),
                std::max(y0, other.y0),
                std::min(x1, other.x1),
                std::min(y1, other.y1)
            };
        }
        simpler_nms_roi_t clamp (simpler_nms_roi_t other) const
        {
            return
            {
                clamp_v(x0, other.x0, other.x1),
                clamp_v(y0, other.y0, other.y1),
                clamp_v(x1, other.x0, other.x1),
                clamp_v(y1, other.y0, other.y1)
            };
        }
    };

    struct simpler_nms_delta_t { Dtype shift_x, shift_y, log_w, log_h; };
    struct simpler_nms_proposal_t { simpler_nms_roi_t roi; Dtype confidence; size_t ord; };

    static std::vector<simpler_nms_roi_t> simpler_nms_perform_nms(
            const std::vector<simpler_nms_proposal_t>& proposals,
            float iou_threshold,
            size_t top_n);

    static void sort_and_keep_at_most_top_n(
            std::vector<simpler_nms_proposal_t>& proposals,
            size_t top_n);

    static simpler_nms_roi_t simpler_nms_gen_bbox(
            const anchor& box,
            const simpler_nms_delta_t& delta,
            int anchor_shift_x,
            int anchor_shift_y);
};

/**
 * @brief Generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * This class is implemented with reference to DetectionOutputLayer
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class FasterRcnnDetectionOutputLayer : public Layer<Dtype> {
 public:
  explicit FasterRcnnDetectionOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FasterRcnnDetectionOutput"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @brief Do non maximum suppression (nms) on prediction results.
   *
   * @param bottom input Blob vector
   *   -# @f$ (C1 \times C2) @f$
   *      bbox_pred: C1 ROIs, C2 = 4 * num of classes
   *   -# @f$ (C1 \times C2) @f$
   *      cls_prob:  C1 ROIs, C2 = 1 * num of classes
   *   -# @f$ (C1 \times C2) @f$
   *      rois:  C1 ROIs, C2 = 5: [0, x0, y0, x1, y1]
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N is the number of detections after nms, and each row is:
   *      [image_id, label, confidence, xmin, ymin, xmax, ymax]
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int num_classes_;
  bool share_location_;
  int num_loc_classes_;
  int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  int keep_top_k_;
  float confidence_threshold_;

  int num_;
  int num_priors_;

  float nms_threshold_;
  int top_k_;
  float eta_;

  Blob<Dtype> bbox_preds_;
  Blob<Dtype> bbox_permute_;
  Blob<Dtype> conf_permute_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
