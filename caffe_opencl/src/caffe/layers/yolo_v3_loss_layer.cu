#include <vector>

#include "caffe/layers/yolo_v3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloV3LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // 将上一层的结果赋值给temp_top
  // copy the bottom[0] result to temp_top
  caffe_cpu_copy<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), temp_top.mutable_cpu_data());
  Dtype *p_top_data = temp_top.mutable_cpu_data();

  // 根据下面公式，修正temp_top的值
  // temp_top的大小为 number_sample * N * M * anchor_size * (4 + 1 + class_number)

  // set the temp_top data
  // 遍历每个样本的预测结果，测试阶段为 1
  for(int b = 0; b < temp_top.shape(0); ++b){

    // 根据anchor_size遍历预测结果
	for(int n = 0; n < anchor_index.size(); ++n){

		// 计算每一个样本预测框的步长，N * M * anchor_size * (4 + 1 + class_number)
		int l_output = temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number) * anchor_index.size();

		// 得到第n个anchor的预测起始点，之后依次是 N * M * 4 的 x, y, w, h
		int index = b * l_output + n * temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number);

		// 为 x, y 增加delta变量，x_new = delta(x) = 1 / (1 + exp(-x))
		for(int i = 0; i < 2 * temp_top.shape(2) * temp_top.shape(3); ++i){
			p_top_data[index + i] = 1.0 / (1.0 + exp(-p_top_data[index + i]));	
		}
		
		// 得到第n个anchor的类别预测起始点，之后依次是 N * M * (1 + class_number) 的 confidence, class_1_prob, class_2_prob ...
		index = b * l_output + n * temp_top.shape(2) * temp_top.shape(3) * (4 + 1 + class_number) + 4 * temp_top.shape(2) * temp_top.shape(3);
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
INSTANTIATE_LAYER_GPU_FUNCS(YoloV3LossLayer);

}  // namespace caffe
