#include <vector>

#include "caffe/layers/yolo_v3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_OPENCV
template <typename Dtype>
cv::Mat drawGroundTruth(cv::Mat img, vector<Box<Dtype> > ground_truth_box){
	int width = img.cols;
	int height = img.rows;
	cv::Mat result_img = img.clone();
	for(int n = 0; n < ground_truth_box.size(); ++n){
		// get class label
        	stringstream stream;  
		stream << ground_truth_box[n].label;
		string label_str = stream.str();
		// resize the rect
		Box<Dtype> b = ground_truth_box[n];
		int left  = (b.x - b.w/2.) * width;
            	int right = (b.x + b.w/2.) * width;
            	int top   = (b.y - b.h/2.) * height;
            	int bot   = (b.y + b.h/2.) * height;

            	if(left < 0) left = 0;
            	if(right > width-1) right = width-1;
            	if(top < 0) top = 0;
            	if(bot > height-1) bot = height-1;
		// draw the rect
		cv::rectangle(result_img, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(0, 0, 255), 2);
		cv::putText(result_img, label_str, cv::Point(left + 5, top + 5), cv::FONT_HERSHEY_COMPLEX, 0.25, cv::Scalar(0, 255, 255));
	}
	return result_img;
}

template <typename Dtype>
cv::Mat drawDetectionResult(cv::Mat img, vector<Detection<Dtype> > single_det, float thresh){
	int width = img.cols;
	int height = img.rows;
	cv::Mat result_img = img.clone();
	for(int n = 0; n < single_det.size(); ++n){
		// get class label
        	stringstream stream;  
		for(int m = 0; m < single_det[n].prob.size(); ++m){
			if(single_det[n].prob[m] > thresh){
				stream << m << ": "<<  single_det[n].prob[m] << " | ";
			}
		}
		string label_str = stream.str();
		if(label_str.size() > 0){
			// resize the rect
			Box<Dtype> b = single_det[n].bbox;
			int left  = (b.x - b.w/2.) * width;
            		int right = (b.x + b.w/2.) * width;
            		int top   = (b.y - b.h/2.) * height;
            		int bot   = (b.y + b.h/2.) * height;

            		if(left < 0) left = 0;
            		if(right > width-1) right = width-1;
            		if(top < 0) top = 0;
            		if(bot > height-1) bot = height-1;
			// draw the rect
			cv::rectangle(result_img, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(0, 0, 255), 2);
			cv::putText(result_img, label_str, cv::Point(left + 5, top + 5), cv::FONT_HERSHEY_COMPLEX, 0.25, cv::Scalar(0, 255, 255));
		}
	}
	return result_img;
}

template <typename Dtype>
bool cmp(Detection<Dtype> a, Detection<Dtype> b){
	float diff = 0;
	if(b.sort_class >= 0){
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	}else{
		diff = a.objectness - b.objectness;
	}
	if(diff < 0){
		return true;
	}else{
		return false;
	}
}

template <typename Dtype>
void nmsSort(vector<Detection<Dtype > > &dets, int class_number, float thresh){
	// remove objectness rect.
	int k = dets.size() - 1;
	for(int i = 0; i < k; ++i){
		if(dets[i].objectness < 1e-6){
			Detection<Dtype> tmp = dets[i];
			dets[i] = dets[k];
			dets[k] = tmp;
			--k;
			--i;
		}
	}	
	int total = k + 1;
	for(k = 0; k < class_number; ++k){
		for(int i = 0; i < total; ++i){
			dets[i].sort_class = k;
		}
		// sorted by class_prob
		std::sort(dets.begin(), dets.begin() + total, cmp<Dtype>);
		for(int i = 0; i < total; ++i){
			if(dets[i].prob[k] < 1e-6){
				continue;
			}
			Box<Dtype> a = dets[i].bbox;
			for(int j = i + 1; j < total; ++j){
				Box<Dtype> b = dets[j].bbox;
				float iou = boxIOU(a, b);
				if(iou > thresh){
					dets[j].prob[k] = 0;
				}
			}
		}	
	}
}

template <typename Dtype>
cv::Mat getOpenCVImage(const Dtype *input_data, const int n_width, const int n_height, const int n_channels){
	cv::Mat image;
	if(n_channels == 1){
		image = cv::Mat::zeros(cv::Size(n_width, n_height), CV_8UC1);
	}else{
		image = cv::Mat::zeros(cv::Size(n_width, n_height), CV_8UC3);
	}
	uchar *iptr = (uchar*)image.data;
	for (int h = 0; h < n_height; ++h){
		for (int w = 0; w < n_width; ++w){
			for (int c = 0; c < n_channels; ++c){
				iptr[h * n_width * n_channels + w * n_channels + c] = input_data[c * n_height * n_width + h * n_width + w]; 
			}
		}
	}
	return image;
}

#endif

int getIndex(const int batch_number, const int location, int skip_channels, int w, int h, int layer_output_number, int class_number){
    int n =   location / (w*h);
    int loc = location % (w*h);
    return batch_number * layer_output_number + n * w * h * (4 + class_number + 1) + skip_channels * w * h + loc;
}

template <typename Dtype>
Box<Dtype> getYoloBox(const Dtype *x, const vector<vector<float> > anchor_boxes, const int anchor_index, const int index, const int i, const int j, const int lw, const int lh, const int w, const int h, const int stride){
    Box<Dtype> b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * anchor_boxes[anchor_index][0] / w;
    b.h = exp(x[index + 3*stride]) * anchor_boxes[anchor_index][1] / h;
    b.label = x[index + 4*stride];
    return b;
}

template <typename Dtype>
Box<Dtype> getGroundTruth(const Dtype *y, const int index, const int stride){
    Box<Dtype> ground_truth;
    ground_truth.x = y[index + 0*stride];
    ground_truth.y = y[index + 1*stride];
    ground_truth.w = y[index + 2*stride];
    ground_truth.h = y[index + 3*stride];
    ground_truth.label = y[index + 4*stride];
    return ground_truth;
}

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2)
{
    Dtype l1 = x1 - w1/2;
    Dtype l2 = x2 - w2/2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1/2;
    Dtype r2 = x2 + w2/2;
    Dtype right = r1 < r2 ? r1 : r2;
    return right - left;
}

template <typename Dtype>
Dtype boxIntersection(Box<Dtype> a, Box<Dtype> b)
{
    Dtype w = overlap(a.x, a.w, b.x, b.w);
    Dtype h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    Dtype area = w*h;
    return area;
}

template <typename Dtype>
Dtype boxUnion(Box<Dtype> a, Box<Dtype> b)
{
    Dtype i = boxIntersection(a, b);
    Dtype u = a.w*a.h + b.w*b.h - i;
    return u;
}

template <typename Dtype>
Dtype boxIOU(Box<Dtype> a, Box<Dtype> b){
    return boxIntersection(a, b)/boxUnion(a, b);
}

template <typename Dtype>
void deltaYoloClass(const Dtype* x, Dtype *y, const int class_index, const int class_idx, const int class_number, const int stride, float *avg_cat){
    if(y[class_index]){
	y[class_index + stride * class_idx] = 1 - x[class_index + stride * class_idx];
	if(avg_cat) *avg_cat += x[class_index + stride * class_idx];
	return;
    }
    for(int n = 0; n < class_number; ++n){
        y[class_index + stride*n] = ((n == class_idx) ? 1 : 0) - x[class_index + stride*n];
	if(n == class_idx && avg_cat) *avg_cat += x[class_index + stride*n];
    }
}

template <typename Dtype>
Dtype deltaYoloBox(Box<Dtype> truth, const Dtype *x, vector<vector<float> > anchor_boxes, int anchor_index, int index, int i, int j, int lw, int lh, int w, int h, Dtype *y, float scale, int stride){
    Box<Dtype> predict_box = getYoloBox(x, anchor_boxes, anchor_index, index, i, j, lw, lh, w, h, stride);
    //printf("predict_box: %f %f %f %f\n", predict_box.x, predict_box.y, predict_box.w, predict_box.h);
    //printf("truth: %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);
    Dtype iou = boxIOU(predict_box, truth);
    //printf("IOU: %f\n", iou);
    Dtype tx = (truth.x*lw - i);
    Dtype ty = (truth.y*lh - j);
    Dtype tw = log(truth.w*w / anchor_boxes[anchor_index][0]);
    Dtype th = log(truth.h*h / anchor_boxes[anchor_index][1]);

    y[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    y[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    y[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    y[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

int getPosInAnchorIndex(vector<int> anchor_index, int index){
	for(int i = 0; i < anchor_index.size(); ++i){
		if(index == anchor_index[i]){
			return i;
		}
	}
	return -1;
}

template <typename Dtype>
void YoloV3LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  // load Yolo params
  YoloParameter yolo_param = this->layer_param_.yolo_param();
  max_box_number = yolo_param.max_box_number();
  ignore_thresh = yolo_param.ignore_thresh();
  truth_thresh = yolo_param.truth_thresh();
  class_number = yolo_param.class_number();
  input_width = yolo_param.input_width();
  input_height = yolo_param.input_height();
  is_debug = yolo_param.is_debug();
  is_show_more_information = yolo_param.is_show_more_information();
  is_train = yolo_param.is_train();
  box_thresh = yolo_param.box_thresh();
  nms_thresh = yolo_param.nms_thresh();
  top_k = yolo_param.top_k();
  yolo_name = this->layer_param_.name();
  const int num_anchor_index = yolo_param.anchor_index_size();
  anchor_index = vector<int>(num_anchor_index, 0);
  for(int i = 0; i < num_anchor_index; ++i){
        anchor_index[i] = yolo_param.anchor_index(i);
  }
  
  const int num_anchor_box = yolo_param.anchor_box_size();
  anchor_boxes = vector<vector<float> >(num_anchor_box, vector<float>(2, 0));
  for(int i = 0; i < num_anchor_box; ++i){
	anchor_boxes[i][0] = yolo_param.anchor_box(i).width();
	anchor_boxes[i][1] = yolo_param.anchor_box(i).height();
  }
  // Test for init parameters.
  LOG(INFO) << "YoloV3LossLayer parameter initialization, "
  	    << " max_box_number: " << max_box_number
            << " ignore_thresh: " << ignore_thresh
            << " truth_thresh: " << truth_thresh 
            << " class_number: " << class_number
            << " input_width: " << input_width
            << " input_height: " << input_height
            << " anchor_index: " << anchor_index.size()
            << " anchor_boxes: " << anchor_boxes.size();
  for(int i = 0; i < num_anchor_index; ++i){
  	LOG(INFO) << "anchor_index " << i << ": " << anchor_index[i];
  }
  for(int i = 0; i < num_anchor_box; ++i){
  	LOG(INFO) << "anchor_boxes " << i << ": [" << anchor_boxes[i][0] << " ," << anchor_boxes[i][1] << "]";
  }
}

template <typename Dtype>
void YoloV3LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // check the anchor box number is match with the label layer.
  CHECK_EQ(anchor_index.size() * (4 + 1 + class_number), bottom[0]->shape(1))
      << "Inputs must have the same dimension.";
  if(is_train){
  	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  	top[0]->Reshape(loss_shape);
  }else{
	vector<int> result_shape;
	int batch_number = bottom[0]->shape(0);
	int top_k_rectangles = top_k;
	// batch * class_number * top_k * (x, y, w, h, obj, class_prob)
	result_shape.push_back(batch_number);
	result_shape.push_back(class_number);
	result_shape.push_back(top_k_rectangles);
	result_shape.push_back(6);
	top[0]->Reshape(result_shape);
  }
  //top[0]->ReshapeLike(*bottom[0]);
  // bottom 1: predict result.  
  diff_.ReshapeLike(*bottom[0]);
  temp_top.ReshapeLike(*bottom[0]);
  
}

template <typename Dtype>
void YoloV3LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // copy the bottom[0] result to temp_top
  caffe_cpu_copy<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), temp_top.mutable_cpu_data());
  Dtype *p_top_data = temp_top.mutable_cpu_data();
  // set the temp_top data
  for(int b = 0; b < temp_top.shape(0); ++b){
	for(int n = 0; n < anchor_index.size(); ++n){
		int l_output = temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number) * anchor_index.size(); 
		int index = b * l_output + n * temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number);
                //printf("b: %d | n: %d | outputs: %d | index: %d ", b, n, l_output, index);
		for(int i = 0; i < 2 * temp_top.shape(2) * temp_top.shape(3); ++i){
			p_top_data[index + i] = 1.0 / (1.0 + exp(-p_top_data[index + i]));	
		}
		index = b * l_output + n * temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number) + 4 * temp_top.shape(2) * temp_top.shape(3);
                //printf(" %d, ", index);
		for(int i = 0; i < (1 + class_number) * temp_top.shape(2) * temp_top.shape(3); ++i){
			p_top_data[index + i] = 1.0 / (1.0 + exp(-p_top_data[index + i]));	
		}
	}
  }
  //for(int kk = 0; kk < temp_top.count(); ++kk){
  //	printf("%f, %f\n", bottom[0]->cpu_data()[kk], temp_top.cpu_data()[kk]);
  //}
  const int top_count = top[0]->count();
  Dtype *top_data = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top_count, Dtype(0), top_data);

  if(is_train){
  	float avg_iou = 0;
  	int count_iou = 0;
  	int class_count = 0;
  	float avg_cat = 0;
  	float avg_obj = 0;
  	float recall = 0;
  	float recall75 = 0;
  	int total_ground_truth_count = 0;
  	// Yolo v3 loss define
  	//const Dtype *p_predict_data = bottom[0]->cpu_data();
  	const Dtype *p_predict_data = temp_top.cpu_data();
  	const Dtype *p_ground_truth_data = bottom[1]->cpu_data();
  	Dtype *p_diff_data = diff_.mutable_cpu_data();
  	const int diff_count = diff_.count();
  	// set diff to zero.
  	caffe_set<Dtype>(diff_count, Dtype(0), p_diff_data);
	
  	int batch_number = bottom[0]->shape(0);
  	int channels = bottom[0]->shape(1);
  	int height = bottom[0]->shape(2);
  	int width = bottom[0]->shape(3);
  	// height * width * anchor_index.size() * (4 + 1 + class_number)
  	int one_sample_predict_number = channels * height * width;
  	//LOG(INFO) << "batch: "<< batch_number << " h: " << height << " w:" << width << " n:" << anchor_index.size();
  	for(int b = 0; b < batch_number; ++b){
		for(int j = 0; j < height; ++j){
			for(int i = 0; i < width; ++i){
				for(int n = 0; n < anchor_index.size(); ++n){
					// get best match iou class for this anchor box.
					// 1: different sample, 2: different anchor boxes, 3: different anchor position.
					int predict_box_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * 0 + j * width + i;
					Box<Dtype> predict_box = getYoloBox(p_predict_data, anchor_boxes, anchor_index[n], predict_box_index, i, j, width, height, input_width, input_height, width * height);
					float best_iou = 0.0f;
					int best_t = 0;
					for(int t = 0; t < max_box_number; ++t){
						int ground_truth_index = b * max_box_number * (4 + 1) + t * (4 + 1);
						Box<Dtype> ground_truth_box = getGroundTruth(p_ground_truth_data, ground_truth_index, 1);
						if(!ground_truth_box.isAvailable()){
							break;
						}
						float iou = boxIOU(predict_box, ground_truth_box);
						if(iou > best_iou){
							best_iou = iou;
							best_t = t;
						}
					}
					// caculate confidence loss.
					int obj_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * 4 + j * width + i;
  					//LOG(INFO) << "box_index: "<< predict_box_index << " obj_index: "<< obj_index;
	
					p_diff_data[obj_index] = 0 - p_predict_data[obj_index];
					if(best_iou > ignore_thresh){
						p_diff_data[obj_index] = 0;
					}
					if(best_iou > truth_thresh){
						p_diff_data[obj_index] = 1 - p_predict_data[obj_index];
		               	 	        int class_idx = p_ground_truth_data[ b * max_box_number * (4 + 1) + best_t*(4 + 1) + 4];
						// two class 1,2 ==> 0,1
						//class_idx = class_idx - 1;
						int predict_class_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * 5 + j * width + i;
						// caculate class diff
						deltaYoloClass(p_predict_data, p_diff_data, predict_class_index, class_idx, class_number, height * width, 0);
						// caculate coordinate diff
						int ground_truth_index = b * max_box_number * (4 + 1) + best_t * (4 + 1);
						Box<Dtype> ground_truth_box = getGroundTruth(p_ground_truth_data, ground_truth_index, 1);
						deltaYoloBox(ground_truth_box, p_predict_data, anchor_boxes, anchor_index[n], predict_box_index, i, j, width, height, input_width, input_height, p_diff_data, (2 - ground_truth_box.w * ground_truth_box.h), width * height);
					}
				}
			}
		}
		for(int t = 0; t < max_box_number; ++t){
			//LOG(INFO) << max_box_number;
  			//LOG(INFO) << "truth_index: "<< b * max_box_number * (4 + 1) + t * (4 + 1) << ", b: " << b << ", truths: " << max_box_number * 5;
			Box<Dtype> ground_truth_box = getGroundTruth(p_ground_truth_data, b * max_box_number * (4 + 1) + t * (4 + 1), 1);
  			//LOG(INFO) << "ground_truth_box_detail "<< t << ": " << ground_truth_box.x << " ," << ground_truth_box.y << " ," << ground_truth_box.w << ", " << ground_truth_box.h << ", " << ground_truth_box.label;
  			//LOG(INFO) << "ground_truth_box_available: " << ground_truth_box.isAvailable();
			if(!ground_truth_box.isAvailable()){
				break;
			}
			total_ground_truth_count += 1;
			float best_iou = 0.0f;
			int best_n = 0;
			// convert ground truth to grid_x, grid_y.
			int i = (ground_truth_box.x * width);
       	     		int j = (ground_truth_box.y * height);
       		     	Box<Dtype> truth_shift = ground_truth_box;
       	     		truth_shift.x = 0;
			truth_shift.y = 0;
  			//LOG(INFO) << "truth_shift: " << truth_shift.x << ", " << truth_shift.y << ", " << truth_shift.w << ", " << truth_shift.h;
      		      	for(int n = 0; n < anchor_boxes.size(); ++n){
               			Box<Dtype> pred;
				pred.x = 0;
				pred.y = 0;
               			pred.w = anchor_boxes[n][0] / input_width;
               			pred.h = anchor_boxes[n][1] / input_height;
                		float iou = boxIOU(pred, truth_shift);
  				//LOG(INFO) << "pred: " << pred.x << ", " << pred.y << ", " << pred.w << ", " << pred.h;
  				//LOG(INFO) << "iou: " << iou << " , max_iou: " << best_iou;
                		if (iou > best_iou){
       	          	   		best_iou = iou;
       	             			best_n = n;
       		         	}
       		     	}
			int anchor_best_index = getPosInAnchorIndex(anchor_index, best_n);
  			//LOG(INFO) << "Anchor_boxes: " << anchor_boxes.size() << ", Best_n: " << best_n << ", Anchor_best_index: " << anchor_best_index;
			if(anchor_best_index >= 0){
				int predict_box_index = b * one_sample_predict_number + anchor_best_index * height * width * (4 + 1 + class_number) + height * width * 0 + j * width + i;
				float iou = deltaYoloBox(ground_truth_box, p_predict_data, anchor_boxes, best_n, predict_box_index, i, j, width, height, input_width, input_height, p_diff_data, (2 - ground_truth_box.w * ground_truth_box.h), width * height);
	

				int obj_index = b * one_sample_predict_number + anchor_best_index * height * width * (4 + 1 + class_number) + height * width * 4 + j * width + i;
				avg_obj += p_predict_data[obj_index];
				p_diff_data[obj_index] = 1 - p_predict_data[obj_index];
				
			        int class_idx = p_ground_truth_data[ b * max_box_number * (4 + 1) + t*(4 + 1) + 4];
				// two class 1,2 ==> 0,1
				//class_idx = class_idx - 1;
				int predict_class_index = b * one_sample_predict_number + anchor_best_index * height * width * (4 + 1 + class_number) + height * width * 5 + j * width + i;
				deltaYoloClass(p_predict_data, p_diff_data, predict_class_index, class_idx, class_number, height * width, &avg_cat);
	
				++count_iou;
                		avg_iou += iou;

       	       		  	++class_count;
       	       		  	if(iou > .5) recall += 1;
       	         		if(iou > .75) recall75 += 1;
			}
		}
  	}

	if(is_show_more_information){
  		LOG(INFO) << "Layer name: " << yolo_name << ", Avg IOU: " << avg_iou / count_iou << ", Class: " << avg_cat / class_count << ", Obj: " << avg_obj / count_iou << ", Recall50: " << recall / count_iou << ", Recall75: " << recall75 / count_iou << ", Count: " << count_iou << ", GT Count: " << total_ground_truth_count;
	}

  	int count = bottom[0]->count();
  	Dtype loss = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  	top[0]->mutable_cpu_data()[0] = loss;

 	 // draw the result.
#ifdef USE_OPENCV
 	 if(is_debug){
  		vector<vector<Detection<Dtype > > > dets = getYoloDetection(box_thresh);
		// do nms
		for(int i = 0; i < dets.size(); ++i){
			nmsSort(dets[i], class_number, nms_thresh);
		}
  		const Dtype *p_ground_truth_img = bottom[2]->cpu_data();
  		int img_channels = bottom[2]->shape(1);
  		int img_height = bottom[2]->shape(2);
  		int img_width = bottom[2]->shape(3);
  		for(int b = 0; b < batch_number; ++b){
			int index = b * img_width * img_height * img_channels;
			cv::Mat img = getOpenCVImage(p_ground_truth_img + index, img_width, img_height, img_channels); 
			cv::Mat draw_img = drawDetectionResult(img, dets[b], box_thresh);
       		 	stringstream stream;
			stream << yolo_name << "_" << b;
			string img_base_name = stream.str();
			string img_name = img_base_name + "_predict.png";
			cv::imwrite(img_name, draw_img);
			vector<Box<Dtype> > ground_truth_box_all;
	
			for(int t = 0; t < max_box_number; ++t){
				Box<Dtype> ground_truth_box = getGroundTruth(p_ground_truth_data, b * max_box_number * (4 + 1) + t * (4 + 1), 1);
				if(!ground_truth_box.isAvailable()){
					break;
				}
				ground_truth_box_all.push_back(ground_truth_box);
			}
			cv::Mat draw_ground_truth_img = drawGroundTruth(img, ground_truth_box_all);
				string img_ground_truth_name = img_base_name + "_ground_truth.png";
			cv::imwrite(img_ground_truth_name, draw_ground_truth_img);
  		}
  	}
	#endif
  }else{
  	vector<vector<Detection<Dtype > > > dets = getYoloDetection(box_thresh);
	// do nms
	for(int i = 0; i < dets.size(); ++i){
		nmsSort(dets[i], class_number, nms_thresh);
	}
 	if(is_debug && bottom.size() == 3){
  		const Dtype *p_ground_truth_img = bottom[2]->cpu_data();
  		const Dtype *p_ground_truth_data = bottom[1]->cpu_data();
		int batch_number = bottom[2]->shape(0);
  		int img_channels = bottom[2]->shape(1);
  		int img_height = bottom[2]->shape(2);
  		int img_width = bottom[2]->shape(3);
  		for(int b = 0; b < batch_number; ++b){
			int index = b * img_width * img_height * img_channels;
			cv::Mat img = getOpenCVImage(p_ground_truth_img + index, img_width, img_height, img_channels); 
			cv::Mat draw_img = drawDetectionResult(img, dets[b], box_thresh);
       		 	stringstream stream;
			stream << yolo_name << "_" << b;
			string img_base_name = stream.str();
			string img_name = img_base_name + "_predict.png";
			cv::imwrite(img_name, draw_img);
			vector<Box<Dtype> > ground_truth_box_all;
	
			for(int t = 0; t < max_box_number; ++t){
				Box<Dtype> ground_truth_box = getGroundTruth(p_ground_truth_data, b * max_box_number * (4 + 1) + t * (4 + 1), 1);
				if(!ground_truth_box.isAvailable()){
					break;
				}
				ground_truth_box_all.push_back(ground_truth_box);
			}
			cv::Mat draw_ground_truth_img = drawGroundTruth(img, ground_truth_box_all);
				string img_ground_truth_name = img_base_name + "_ground_truth.png";
			cv::imwrite(img_ground_truth_name, draw_ground_truth_img);

		}
	}
  	Dtype *p_result_detection = top[0]->mutable_cpu_data();
	int dect_param_number = top[0]->shape(3);
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
}

template <typename Dtype>
void YoloV3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // caculate the sum loss
  //int count = bottom[0]->count();
  //Dtype loss = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //LOG(INFO) << "Yolo v3 loss: " << loss;
  // set diff_ to bias given.
  if(is_train){
  	for (int i = 0; i < 2; ++i) {
    		//LOG(INFO) << "Propagate: " << i << ", value: " << propagate_down[i];
    		if (propagate_down[i]) {
      			//const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      			const Dtype alpha = (i == 0) ? -1 : 1;
      			caffe_cpu_axpby(
          		bottom[i]->count(),              // count
          		alpha,                              // alpha
          		diff_.cpu_data(),                   // a
          		Dtype(0),                           // beta
          		bottom[i]->mutable_cpu_diff());  // b
    		}
  	}
  }
}

template <typename Dtype>
vector<vector<Detection<Dtype> > > YoloV3LossLayer<Dtype>::getYoloDetection(float thresh){
  const Dtype *p_predict_data = temp_top.cpu_data();
  int batch_number = temp_top.shape(0);
  int channels = temp_top.shape(1);
  int height = temp_top.shape(2);
  int width = temp_top.shape(3);
  int one_sample_predict_number = channels * height * width;

  vector<vector<Detection<Dtype> > > dets;
  for(int b = 0; b < batch_number; ++b){
	vector<Detection<Dtype> > single_dets;
	for(int j = 0; j < height; ++j){
		for(int i = 0; i < width; ++i){
			for(int n = 0; n < anchor_index.size(); ++n){
				int obj_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * 4 + j * width + i;
				float objectness = p_predict_data[obj_index];
				//printf("objectness: %f\n", objectness);
				if(objectness <= thresh){
					continue;
				}
				//printf("objectness: %f\n", objectness);

				Detection<Dtype> det;
				int predict_box_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * 0 + j * width + i;
				Box<Dtype> predict_box = getYoloBox(p_predict_data, anchor_boxes, anchor_index[n], predict_box_index, i, j, width, height, input_width, input_height, width * height);
				det.bbox = predict_box;
				det.classes = class_number;	
				det.objectness = objectness;
				det.prob = vector<float>(class_number, 0);
				for(int kk = 0; kk < class_number; ++kk){
					int class_index = b * one_sample_predict_number + n * height * width * (4 + 1 + class_number) + height * width * (4 + 1 + kk) + j * width + i;
					float prob = objectness * p_predict_data[class_index];
					/*
					if(is_debug){
						printf("class_number: %d, prob: %f\n", kk, prob);
					}
					*/
					det.prob[kk] = (prob > thresh) ? prob : 0;
				}
				single_dets.push_back(det);
			}
		}
	}
	dets.push_back(single_dets);
  }
  return dets;
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(YoloV3LossLayer, Forward);
#endif
INSTANTIATE_CLASS(YoloV3LossLayer);
REGISTER_LAYER_CLASS(YoloV3Loss);

}  // namespace caffe
