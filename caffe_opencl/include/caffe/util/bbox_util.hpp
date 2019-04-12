#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

typedef map<int_tp, vector<NormalizedBBox> > LabelBBox;
typedef EmitConstraint_EmitType EmitType;
typedef PriorBoxParameter_CodeType CodeType;
typedef MultiBoxLossParameter_MatchType MatchType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_ConfLossType ConfLossType;
typedef MultiBoxLossParameter_MiningType MiningType;

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true);
// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in ascend order based on the score value.
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in descend order based on the score value.
bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);

// Generate unit bbox [0, 0, 1, 1]
NormalizedBBox UnitBBox();

// Check if a bbox is cross boundary or not.
bool IsCrossBoundaryBBox(const NormalizedBBox& bbox);

// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true);
void CumSum(const vector<pair<float, int_tp> >& pairs, vector<int_tp>* cumsum);
                     
template <typename Dtype>
void setNormalizedBBox(NormalizedBBox& bbox, Dtype x, Dtype y, Dtype w, Dtype h)
{
  Dtype xmin = x - w/2.0;
  Dtype xmax = x + w/2.0;
  Dtype ymin = y - h/2.0;
  Dtype ymax = y + h/2.0;

  if (xmin < 0.0){
    xmin = 0.0;
  }
  if (xmax > 1.0){
    xmax = 1.0;
  }
  if (ymin < 0.0){
    ymin = 0.0;
  }
  if (ymax > 1.0){
    ymax = 1.0;
  }  
  bbox.set_xmin(xmin);
  bbox.set_ymin(ymin);
  bbox.set_xmax(xmax);
  bbox.set_ymax(ymax);
  float bbox_size = BBoxSize(bbox, true);
  bbox.set_size(bbox_size);
}

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int_tp num_det,
      map<int_tp, LabelBBox>* all_detections) {
  all_detections->clear();
  for (int_tp i = 0; i < num_det; ++i) {
    int_tp start_idx = i * 7;
    int_tp item_id = det_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int_tp label = det_data[start_idx + 1];
    NormalizedBBox bbox;
    Dtype x = det_data[start_idx + 3];
    Dtype y = det_data[start_idx + 4];
    Dtype w = det_data[start_idx + 5];
    Dtype h = det_data[start_idx + 6];

    setNormalizedBBox(bbox, x, y, w, h);
    bbox.set_score(det_data[start_idx + 2]); //confidence   
    (*all_detections)[item_id][label].push_back(bbox);
  }
}
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int_tp num_gt,
      map<int_tp, LabelBBox >* all_gt_bboxes) {
  all_gt_bboxes->clear();
  int_tp cnt = 0;
  for (int_tp t = 0; t < 30; ++t){
    vector<Dtype> truth;
    int_tp label = gt_data[t * 5];
    Dtype x = gt_data[t * 5 + 1];
    Dtype y = gt_data[t * 5 + 2];
    Dtype w = gt_data[t * 5 + 3];
    Dtype h = gt_data[t * 5 + 4];

    if (!w) break;
    cnt++;
    int_tp item_id = 0;
    NormalizedBBox bbox;
    setNormalizedBBox(bbox, x, y, w, h);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
}
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype l1 = x1 - w1/2;
  Dtype l2 = x2 - w2/2;
  Dtype left = l1 > l2 ? l1 : l2;
  Dtype r1 = x1 + w1/2;
  Dtype r2 = x2 + w2/2;
  Dtype right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  NormalizedBBox Bbox1, Bbox2;
  setNormalizedBBox(Bbox1, box[0], box[1], box[2], box[3]);
  setNormalizedBBox(Bbox2, truth[0], truth[1], truth[2], truth[3]);
  return JaccardOverlap(Bbox1, Bbox2, true);
}

template <typename Dtype>
void disp(Blob<Dtype>& swap)
{
  std::cout<<"#######################################"<<std::endl;
  for (int_tp b = 0; b < swap.num(); ++b)
    for (int_tp c = 0; c < swap.channels(); ++c)
      for (int_tp h = 0; h < swap.height(); ++h)
      {
  	std::cout<<"[";
        for (int_tp w = 0; w < swap.width(); ++w)
	{
	  std::cout<<swap.data_at(b,c,h,w)<<",";	
	}
	std::cout<<"]"<<std::endl;
      }
  return;
}


template <typename Dtype>
class PredictionResult{
  public:
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
    Dtype objScore;
    Dtype classScore;
    Dtype confidence;
    int_tp classType;
};
template <typename Dtype>
void class_index_and_score(Dtype* input, int_tp classes, PredictionResult<Dtype>& predict)
{
  Dtype sum = 0;
  Dtype large = input[0];
  int_tp classIndex = 0;
  for (int_tp i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int_tp i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  
  for (int_tp i = 0; i < classes; ++i){
    input[i] = input[i] / sum;   
  }
  large = input[0];
  classIndex = 0;

  for (int_tp i = 0; i < classes; ++i){
    if (input[i] > large){
      large = input[i];
      classIndex = i;
    }
  }  
  predict.classType = classIndex ;
  predict.classScore = large;
}
template <typename Dtype>
void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, vector<Dtype> biases, int_tp n, int_tp index, int_tp i, int_tp j, int_tp w, int_tp h){
  predict.x = (i + sigmoid(x[index + 0])) / w;
  predict.y = (j + sigmoid(x[index + 1])) / h;
  predict.w = exp(x[index + 2]) * biases[2*n] / w;
  predict.h = exp(x[index + 3]) * biases[2*n+1] / h;
}
template <typename Dtype>
void ApplyNms(vector< PredictionResult<Dtype> >& boxes, vector<int_tp>& idxes, Dtype threshold) {
  //map<int_tp, int_tp> idx_map; 
  vector<int_tp> idx_map(boxes.size(), 0);
  for (int_tp i = 0; i < boxes.size() - 1; ++i) {
    //if (idx_map.find(i) != idx_map.end()) {
    if (idx_map[i] == 1) {
      continue;
    }
    for (int_tp j = i + 1; j < boxes.size(); ++j) {
      //if (idx_map.find(j) != idx_map.end()) {	
      if (idx_map[j] == 1) {
        continue;
      }
      NormalizedBBox Bbox1, Bbox2;
      setNormalizedBBox(Bbox1, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
      setNormalizedBBox(Bbox2, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h);

      float overlap = JaccardOverlap(Bbox1, Bbox2, true);

      if (overlap >= threshold)
       	idx_map[j] = 1;
    }
  }
  for (int_tp i = 0; i < boxes.size(); ++i) {
  	//if (idx_map.find(i) == idx_map.end()) {
    if (idx_map[i] == 0) {
      idxes.push_back(i);
    }
  }
}

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int_tp>& pair1,
                                   const pair<float, int_tp>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int_tp, int_tp> >& pair1,
                                   const pair<float, pair<int_tp, int_tp> >& pair2);
              
typedef MultiBoxLossParameter_MatchType MatchType;
// Output the real bbox in the original image space.
void OutputBBox(const NormalizedBBox& bbox, const int height, const int width,
                const bool clip, NormalizedBBox* outbbox);
// Clip the NormalizedBBox such that the range for each corner is [0, 1].
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox);

// Clip the bbox such that the bbox is within [0, 0; width, height].
void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox);

// Scale the NormalizedBBox w.r.t. height and width.
void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox);

// Output predicted bbox on the actual image.
void OutputBBox(const NormalizedBBox& bbox, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParameter& resize_param,
                NormalizedBBox* out_bbox);

// Locate bbox in the coordinate system that src_bbox sits.
void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox);

// Project bbox onto the coordinate system defined by src_bbox.
bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox);

// Extrapolate the transformed bbox if height_scale and width_scale is
// explicitly provided, and it is only effective for FIT_SMALL_SIZE case.
void ExtrapolateBBox(const ResizeParameter& param, const int height,
    const int width, const NormalizedBBox& crop_bbox, NormalizedBBox* bbox);

// Compute the coverage of bbox1 by bbox2.
float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

// Encode a bbox according to a prior bbox.
void EncodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool encode_variance_in_target, const NormalizedBBox& bbox,
    NormalizedBBox* encode_bbox);

// Check if a bbox meet emit constraint w.r.t. src_bbox.
bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
    const NormalizedBBox& bbox, const EmitConstraint& emit_constraint);

// Decode a bbox according to a prior bbox.
void DecodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool variance_encoded_in_target, const bool clip_bbox, const float clip_w, const float clip_h,
    const NormalizedBBox& bbox, NormalizedBBox* decode_bbox);

// Decode a set of bboxes according to a set of prior bboxes.
void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const float clip_w, const float clip_h, const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_pred,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, const float clip_w, const float clip_h, vector<LabelBBox>* all_decode_bboxes);

// Match prediction bboxes with ground truth bboxes.
void MatchBBox(const vector<NormalizedBBox>& gt,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    const bool ignore_cross_boundary_bbox,
    vector<int>* match_indices, vector<float>* match_overlaps);

// Find matches between prediction bboxes and ground truth bboxes.
//    all_loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
//    all_gt_bboxes: stores ground truth bboxes for the batch.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
//    multibox_loss_param: stores the parameters for MultiBoxLossLayer.
//    all_match_overlaps: stores jaccard overlaps between predictions and gt.
//    all_match_indices: stores mapping between predictions and ground truth.
void FindMatches(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices);

// Count the number of matches from the match indices.
int CountNumMatches(const vector<map<int, vector<int> > >& all_match_indices,
                    const int num);

// Mine the hard examples from the batch.
//    conf_blob: stores the confidence prediction.
//    all_loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
//    all_gt_bboxes: stores ground truth bboxes for the batch.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
//    all_match_overlaps: stores jaccard overlap between predictions and gt.
//    multibox_loss_param: stores the parameters for MultiBoxLossLayer.
//    all_match_indices: stores mapping between predictions and ground truth.
//    all_loc_loss: stores the confidence loss per location for each image.
template <typename Dtype>
void MineHardExamples(const Blob<Dtype>& conf_blob,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);

// Retrieve bounding box ground truth from gt_data.
//    gt_data: 1 x 1 x num_gt x 7 blob.
//    num_gt: the number of ground truth.
//    background_label_id: the label for background class which is used to do
//      santity check so that no ground truth contains it.
//    all_gt_bboxes: stores ground truth for each image. Label of each bbox is
//      stored in NormalizedBBox.
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);
// Store ground truth bboxes of same label in a group.
template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);

// Get location predictions from loc_data.
//    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_loc_classes: number of location classes. It is 1 if share_location is
//      true; and is equal to number of classes needed to predict otherwise.
//    share_location: if true, all classes share the same location prediction.
//    loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

// Encode the localization prediction and ground truth for each matched prior.
//    all_loc_preds: stores the location prediction, where each item contains
//      location prediction for an image.
//    all_gt_bboxes: stores ground truth bboxes for the batch.
//    all_match_indices: stores mapping between predictions and ground truth.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
//    multibox_loss_param: stores the parameters for MultiBoxLossLayer.
//    loc_pred_data: stores the location prediction results.
//    loc_gt_data: stores the encoded location ground truth.
template <typename Dtype>
void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      Dtype* loc_pred_data, Dtype* loc_gt_data);

// Compute the localization loss per matched prior.
//    loc_pred: stores the location prediction results.
//    loc_gt: stores the encoded location ground truth.
//    all_match_indices: stores mapping between predictions and ground truth.
//    num: number of images in the batch.
//    num_priors: total number of priors.
//    loc_loss_type: type of localization loss, Smooth_L1 or L2.
//    all_loc_loss: stores the localization loss for all priors in a batch.
template <typename Dtype>
void ComputeLocLoss(const Blob<Dtype>& loc_pred, const Blob<Dtype>& loc_gt,
      const vector<map<int, vector<int> > >& all_match_indices,
      const int num, const int num_priors, const LocLossType loc_loss_type,
      vector<vector<float> >* all_loc_loss);

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    conf_preds: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_scores);

// Get confidence predictions from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    class_major: if true, data layout is
//      num x num_classes x num_preds_per_class; otherwise, data layerout is
//      num x num_preds_per_class * num_classes.
//    conf_preds: stores the confidence prediction, where each item contains
//      confidence prediction for an image.
template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const bool class_major, vector<map<int, vector<float> > >* conf_scores);

// Compute the confidence loss for each prior from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    background_label_id: it is used to skip selecting max scores from
//      background class.
//    loss_type: compute the confidence loss according to the loss type.
//    all_match_indices: stores mapping between predictions and ground truth.
//    all_gt_bboxes: stores ground truth bboxes from the batch.
//    all_conf_loss: stores the confidence loss per location for each image.
template <typename Dtype>
void ComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);

// Compute the negative confidence loss for each prior from conf_data.
//    conf_data: num x num_preds_per_class * num_classes blob.
//    num: the number of images.
//    num_preds_per_class: number of predictions per class.
//    num_classes: number of classes.
//    background_label_id: it is used to skip selecting max scores from
//      background class.
//    loss_type: compute the confidence loss according to the loss type.
//    all_conf_loss: stores the confidence loss per location for each image.
template <typename Dtype>
void ComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      vector<vector<float> >* all_conf_loss);

// Encode the confidence predictions and ground truth for each matched prior.
//    conf_data: num x num_priors * num_classes blob.
//    num: number of images.
//    num_priors: number of priors (predictions) per image.
//    multibox_loss_param: stores the parameters for MultiBoxLossLayer.
//    all_match_indices: stores mapping between predictions and ground truth.
//    all_neg_indices: stores the indices for negative samples.
//    all_gt_bboxes: stores ground truth bboxes for the batch.
//    conf_pred_data: stores the confidence prediction results.
//    conf_gt_data: stores the confidence ground truth.
template <typename Dtype>
void EncodeConfPrediction(const Dtype* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      Dtype* conf_pred_data, Dtype* conf_gt_data);

// Get prior bounding boxes from prior_data.
//    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
//    num_priors: number of priors.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

template <typename Dtype>
void GetRoiBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes);

// Get detection results from det_data.
//    det_data: 1 x 1 x num_det x 7 blob.
//    num_det: the number of detections.
//    background_label_id: the label for background class which is used to do
//      santity check so that no detection contains it.
//    all_detections: stores detection results for each class from each image.
template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, LabelBBox>* all_detections);

// Get top_k scores with corresponding indices.
//    scores: a set of scores.
//    indices: a set of corresponding indices.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetTopKScoreIndex(const vector<float>& scores, const vector<int>& indices,
      const int top_k, vector<pair<float, int> >* score_index_vec);

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);

// Get max scores with corresponding indices.
//    scores: an array of scores.
//    num: number of total scores in the array.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >& score_index_vec);

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);

// Do non maximum suppression given bboxes and scores.
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    threshold: the threshold used in non maximum suppression.
//    top_k: if not -1, keep at most top_k picked indices.
//    reuse_overlaps: if true, use and update overlaps; otherwise, always
//      compute overlap.
//    overlaps: a temp place to optionally store the overlaps between pairs of
//      bboxes if reuse_overlaps is true.
//    indices: the kept indices of bboxes after nms.
void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, const bool reuse_overlaps,
      map<int, map<int, float> >* overlaps, vector<int>* indices);

void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, vector<int>* indices);

void ApplyNMS(const bool* overlapped, const int num, vector<int>* indices);

// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    eta: adaptation rate for nms threshold (see Piotr's paper).
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices);

// Do non maximum suppression based on raw bboxes and scores data.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: an array of bounding boxes.
//    scores: an array of corresponding confidences.
//    num: number of total boxes/confidences in the array.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    eta: adaptation rate for nms threshold (see Piotr's paper).
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>& indices);

// Compute cumsum of a set of pairs.
void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);

#ifndef CPU_ONLY  // GPU
#ifdef USE_CUDA
template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
                                      const bool normalized = true);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
                                            const Dtype* bbox2);
#endif //USE_CUDA
template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox,  const Dtype clip_w, const Dtype clip_h, Dtype* bbox_data);

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data);

template <typename Dtype>
void PermuteData24GPU(const int nthreads,
		  const Dtype* data, const int num_channels, const int num_height,
		  const int num_width, Dtype* new_data);

template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int outer_num, const int channels,
    const int inner_num, Dtype* prob);

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data);

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
#ifdef USE_CUDA
template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob);

template <typename Dtype>
void ComputeConfLossGPU(const Blob<Dtype>& conf_blob, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);
#endif // USE_CUDA
#endif  // !CPU_ONLY

#ifdef USE_OPENCV
vector<cv::Scalar> GetColors(const int n);

template <typename Dtype>
void VisualizeBBox(const vector<cv::Mat>& images, const Blob<Dtype>* detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file, bool is_yolo = false);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
