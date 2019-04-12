//#include <cuda_runtime.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "boost/algorithm/string.hpp"
#include "head.h"

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

#define KEYPOINTS_NUMBER 68

DEFINE_int32(step, 9,
	"Step 1 detection.\n"
	"Step 2 keypoints.\n"
	"Step 3 detection and keypoints.\n");
DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones\n");
DEFINE_string(suffix, "png",
	"The suffix of the image.\n");

DEFINE_bool(is_use_gpu, true, "Wether use gpu or not.");
DEFINE_int32(max_people, 10, "max number of people in one image.");

// detection
DEFINE_int32(resize_width_detection, 160, "Width images are resized to");
DEFINE_int32(resize_height_detection, 120, "Height images are resized to");
DEFINE_string(model_dir_detection, "./models/face/", "The model folder.");
DEFINE_string(model_prototxt_detection, "a.prototxt", "The model prototxt.");
DEFINE_string(model_params_detection, "a.h5", "The model params file.");
DEFINE_bool(is_use_label_file_detection, false, "Wether has label or not.");
DEFINE_string(output_name_detection, "F:/download/test_detection.png", "output_name.");

// keypoints
DEFINE_int32(resize_width_keypoints, 240, "Width images are resized to");
DEFINE_int32(resize_height_keypoints, 240, "Height images are resized to");
DEFINE_string(model_dir_keypoints, "./models/face/", "The model folder.");
DEFINE_string(model_prototxt_keypoints, "b.prototxt", "The model prototxt.");
DEFINE_string(model_params_keypoints, "b.h5", "The model params file.");
DEFINE_bool(is_use_label_file_keypoints, false, "Wether has label or not.");
DEFINE_string(output_name_keypoints, "F:/download/test_keypoints.png", "output_name.");
DEFINE_double(keypoint_thred, 0.4, "output_name.");

// test time
DEFINE_int32(iterations, 50, "Test net time cost.");

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
	int count = 0;
#ifndef CPU_ONLY
	count = Caffe::EnumerateDevices(true);
#else
	NO_GPU;
#endif
	for (int i = 0; i < count; ++i) {
		gpus->push_back(i);
	}
}

Net<float>* initCaffeNet(std::string caffe_net_prototxt_file, std::string caffe_net_params_file, bool is_use_gpu) {
	/*
	NetParameter net_prototxt;

	ReadNetParamsFromTextFileOrDie(caffe_net_prototxt_file, &net_prototxt);
	//NetParameter net_param;
	//ReadNetParamsFromBinaryFileOrDie(caffe_net_params_file, &net_param);
	Caffe::SetDevice(0);
	if (is_use_gpu) {
		std::cout << "Using GPU" << std::endl;
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		std::cout << "Using CPU" << std::endl;
		Caffe::set_mode(Caffe::CPU);
	}

	//Net<float> *caffe_net = new Net<float>(net_prototxt);
	//Net<float> *caffe_net = new Net<float>(net_prototxt, Caffe::GetDefaultDevice());
	*/
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() == 0 || !is_use_gpu) {
		std::cout << "Using CPU" << std::endl;
		Caffe::set_mode(Caffe::CPU);
		NetParameter net_prototxt;

		ReadNetParamsFromTextFileOrDie(caffe_net_prototxt_file, &net_prototxt);
		Net<float> *caffe_net = new Net<float>(net_prototxt, Caffe::GetDefaultDevice());
		caffe_net->CopyTrainedLayersFromHDF5(caffe_net_params_file);
		return caffe_net;
	}
	else {
		std::cout << "Use GPU with device ID " << gpus[0] << std::endl;
		//Caffe::SetDevices(gpus);
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpus[0]);

		Net<float> *caffe_net = new Net<float>(caffe_net_prototxt_file, caffe::Phase::TEST, Caffe::GetDefaultDevice());

		caffe_net->CopyTrainedLayersFromHDF5(caffe_net_params_file);
		return caffe_net;

	}
}

bool getInputVec(cv::Mat src_image, vector<Blob<float> *> &input_vec) {
	cv::Mat image;
	src_image.convertTo(image, CV_8UC3);
	int			n_width = image.cols;//IMAGEWIDHT;
	int			n_height = image.rows;//IMAGEHEIGHT;
	int			n_channels = image.channels();
	Blob<float>	*blob_data = new Blob<float>(1, n_channels, n_height, n_width);
	float		*input_data = blob_data->mutable_cpu_data();

	uchar *iptr = (uchar*)image.data;
	for (int h = 0; h < n_height; ++h) {
		for (int w = 0; w < n_width; ++w) {
			for (int c = 0; c < n_channels; ++c) {
				input_data[c * n_height * n_width + h * n_width + w] = iptr[h * n_width * n_channels + w * n_channels + c];
			}
		}
	}

	input_vec.push_back(blob_data);
	return true;
}

// #######################################################
// detection predict
struct Rectangle
{
	float x;
	float y;
	float width;
	float height;
	Rectangle(float x_, float y_, float width_, float height_) {
		x = x_;
		y = y_;
		width = width_;
		height = height_;
	}
};

Rectangle resizeNormaRectangle(float x, float y, float w, float h, int width, int height, int type = 1, float ratio = 1.4) {
	if (type == 0) {
		int left = (x - w / 2.) * width;
		int right = (x + w / 2.) * width;
		int top = (y - h / 2.) * height;
		int bot = (y + h / 2.) * height;

		if (left < 0) left = 0;
		if (right > width - 1) right = width - 1;
		if (top < 0) top = 0;
		if (bot > height - 1) bot = height - 1;
		return Rectangle(left, top, right - left, bot - top);
	}
	else {
		int x_center = x * width;
		int y_center = y * height;
		int det_width = w * width * ratio;
		int det_height = h * height * ratio;
		int max_edge = det_width > det_height ? det_width : det_height;
		int left = x_center - max_edge / 2.;
		int right = x_center + max_edge / 2.;
		int top = y_center - max_edge / 2.;
		int bot = y_center + max_edge / 2.;

		if (left < 0) left = 0;
		if (right > width - 1) right = width - 1;
		if (top < 0) top = 0;
		if (bot > height - 1) bot = height - 1;
		return Rectangle(left, top, right - left, bot - top);
	}
}

Rectangle resizeSrcRectangle(Rectangle resize_rectangle, int resize_width, int resize_height, int img_width, int img_height) {
	Rectangle src_rectangle(0, 0, 0, 0);
	float ratio_width = float(resize_width) / img_width;
	float ratio_height = float(resize_height) / img_height;

	if (ratio_width == ratio_height) {
		int x1 = resize_rectangle.x;
		int y1 = resize_rectangle.y;
		int x2 = resize_rectangle.x + resize_rectangle.width;
		int y2 = resize_rectangle.y + resize_rectangle.height;

		int x1_new = (x1 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y1_new = (y1 - resize_height / 2.) / ratio_height + img_height / 2.;
		int x2_new = (x2 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y2_new = (y2 - resize_height / 2.) / ratio_height + img_height / 2.;

		src_rectangle.x = x1_new;
		src_rectangle.y = y1_new;
		src_rectangle.width = x2_new - x1_new;
		src_rectangle.height = y2_new - y1_new;

	}
	else if (ratio_width < ratio_height) {
		int new_resize_height = int(float(resize_width) / img_width * img_height);
		int bias_h = (resize_height - new_resize_height) / 2.0;
		int x1 = resize_rectangle.x;
		int y1 = resize_rectangle.y - bias_h;
		int x2 = resize_rectangle.x + resize_rectangle.width;
		int y2 = resize_rectangle.y + resize_rectangle.height - bias_h;

		int x1_new = (x1 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y1_new = (y1 - new_resize_height / 2.) / ratio_width + img_height / 2.;
		int x2_new = (x2 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y2_new = (y2 - new_resize_height / 2.) / ratio_width + img_height / 2.;

		src_rectangle.x = x1_new;
		src_rectangle.y = y1_new;
		src_rectangle.width = x2_new - x1_new;
		src_rectangle.height = y2_new - y1_new;

	}
	else {
		int new_resize_width = int(float(resize_height) / img_height * img_width);
		int bias_w = (resize_width - new_resize_width) / 2.0;

		int x1 = resize_rectangle.x - bias_w;
		int y1 = resize_rectangle.y;
		int x2 = resize_rectangle.x + resize_rectangle.width - bias_w;
		int y2 = resize_rectangle.y + resize_rectangle.height;

		int x1_new = (x1 - new_resize_width / 2.) / ratio_height + img_width / 2.;
		int y1_new = (y1 - resize_height / 2.) / ratio_height + img_height / 2.;
		int x2_new = (x2 - new_resize_width / 2.) / ratio_height + img_width / 2.;
		int y2_new = (y2 - resize_height / 2.) / ratio_height + img_height / 2.;

		src_rectangle.x = x1_new;
		src_rectangle.y = y1_new;
		src_rectangle.width = x2_new - x1_new;
		src_rectangle.height = y2_new - y1_new;

	}
	return src_rectangle;
}

cv::Mat createResizeImage(cv::Mat img, const int &resize_width, const int &resize_height) {
	int img_height = img.rows;
	int img_width = img.cols;
	float ratio_width = float(resize_width) / img_width;
	float ratio_height = float(resize_height) / img_height;
	cv::Mat result = cv::Mat::zeros(Size(resize_width, resize_height), img.type());
	if (ratio_width == ratio_height) {
		cv::resize(img, result, Size(resize_width, resize_height));
	}
	else if (ratio_width < ratio_height) {
		// resize width first
		int new_resize_height = int(float(resize_width) / img_width * img_height);
		cv::Mat temp = cv::Mat::zeros(Size(resize_width, new_resize_height), img.type());
		cv::resize(img, temp, Size(resize_width, new_resize_height));
		int bias_h = (resize_height - new_resize_height) / 2.0;
		for (int h = 0; h < new_resize_height; ++h) {
			for (int w = 0; w < resize_width; ++w) {
				if (img.channels() == 3) {
					result.at<Vec3b>(h + bias_h, w) = temp.at<Vec3b>(h, w);
				}
				else {
					result.at<uchar>(h + bias_h, w) = temp.at<uchar>(h, w);
				}
			}
		}
	}
	else {
		// resize height first
		int new_resize_width = int(float(resize_height) / img_height * img_width);
		cv::Mat temp = cv::Mat::zeros(Size(new_resize_width, resize_height), img.type());
		cv::resize(img, temp, Size(new_resize_width, resize_height));
		int bias_w = (resize_width - new_resize_width) / 2.0;
		for (int h = 0; h < resize_height; ++h) {
			for (int w = 0; w < new_resize_width; ++w) {
				if (img.channels() == 3) {
					result.at<Vec3b>(h, w + bias_w) = temp.at<Vec3b>(h, w);
				}
				else {
					result.at<uchar>(h, w + bias_w) = temp.at<uchar>(h, w);
				}
			}
		}

	}
	return result;
}

std::vector<Rectangle> predictFaceDetection(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_detection, int resize_height_detection, int max_people = 1) {
	std::vector<Rectangle> face_rectangles;
	cv::Mat image = createResizeImage(src_image, resize_width_detection, resize_height_detection);
	vector<Blob<float>*>	input_vec;
	vector<Blob<float>*>	output_vec;
	if (!getInputVec(image, input_vec)) {
		return face_rectangles;
	}

	double time0 = cvGetTickCount();
	output_vec = caffe_net->Forward(input_vec);
	double time1 = cvGetTickCount();

	float time_cost = (time1 - time0) / cv::getTickFrequency() * 1000;
	std::cout << "Total cost time : " << time_cost << std::endl;
	int		n_outCount = output_vec[0]->count();
	int		n_num = output_vec[0]->num();
	int		n_width = output_vec[0]->width();
	int		n_height = output_vec[0]->height();
	int		n_channels = output_vec[0]->channels();
	float	*output_maps = new float[n_num * n_width * n_height * n_channels];

	for (int i = 0; i < output_vec.size(); i++) {
		memcpy(output_maps, output_vec[i]->cpu_data(), sizeof(float)*n_outCount);
		for (int i = 0; i < n_num; ++i) {
			for (int c = 0; c < 1; ++c) {
				for (int hh = 0; hh < n_height; ++hh) {
					int index = i * n_channels * n_height * n_width + c * n_height * n_width + hh * n_width;
					float x = output_maps[index + 0];
					float y = output_maps[index + 1];
					float w = output_maps[index + 2];
					float h = output_maps[index + 3];
					float objectness = output_maps[index + 4];
					float prob = output_maps[index + 5];
					//printf("index: %d, x: %f, y: %f, w: %f, h: %f, objectness: %f, prob: %f\n", index, x, y, w, h, objectness, prob);
					// only save prob > 0.5 detection.
					if (objectness > 0.5 && prob > 0.5) {
						Rectangle new_rectangle = resizeNormaRectangle(x, y, w, h, resize_width_detection, resize_height_detection);
						//temp_rect.push_back(new_rectangle);
						// only match 1 people.
						if (face_rectangles.size() < max_people) {
							face_rectangles.push_back(resizeSrcRectangle(new_rectangle, resize_width_detection, resize_height_detection, src_image.cols, src_image.rows));
						}
					}
				}
			}
		}
	}

	delete[] output_maps;
	for (int i = 0; i < input_vec.size(); ++i) {
		if (input_vec[i] != NULL) {
			delete input_vec[i];
		}
	}
	return face_rectangles;
}
//#######################################################################

// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
// keypoints

Point3f resizeSrcPoint3f(Point3f point, int resize_width, int resize_height, int img_width, int img_height) {
	Point3f src_point(0, 0, 0);
	float ratio_width = float(resize_width) / img_width;
	float ratio_height = float(resize_height) / img_height;

	if (ratio_width == ratio_height) {
		int x1 = point.x;
		int y1 = point.y;

		int x1_new = (x1 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y1_new = (y1 - resize_height / 2.) / ratio_height + img_height / 2.;

		src_point.x = x1_new;
		src_point.y = y1_new;
		src_point.z = point.z;
	}
	else if (ratio_width < ratio_height) {
		int new_resize_height = int(float(resize_width) / img_width * img_height);
		int bias_h = (resize_height - new_resize_height) / 2.0;
		int x1 = point.x;
		int y1 = point.y - bias_h;

		int x1_new = (x1 - resize_width / 2.) / ratio_width + img_width / 2.;
		int y1_new = (y1 - new_resize_height / 2.) / ratio_width + img_height / 2.;
		src_point.x = x1_new;
		src_point.y = y1_new;
		src_point.z = point.z;
	}
	else {
		int new_resize_width = int(float(resize_height) / img_height * img_width);
		int bias_w = (resize_width - new_resize_width) / 2.0;
		int x1 = point.x - bias_w;
		int y1 = point.y;
		int x1_new = (x1 - new_resize_width / 2.) / ratio_height + img_width / 2.;
		int y1_new = (y1 - resize_height / 2.) / ratio_height + img_height / 2.;
		src_point.x = x1_new;
		src_point.y = y1_new;
		src_point.z = point.z;
	}
	return src_point;
}

vector<cv::Point3f> predictFaceKeypoints(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_keypoints, int resize_height_keypoints, int max_keypoints= KEYPOINTS_NUMBER){
	vector<cv::Point3f> face_keypoints;
	cv::Mat image = createResizeImage(src_image, resize_width_keypoints, resize_height_keypoints);
	vector<Blob<float>*>	input_vec;
	vector<Blob<float>*>	output_vec;
	if (!getInputVec(image, input_vec)) {
		return face_keypoints;
	}

	double time0 = cvGetTickCount();
	output_vec = caffe_net->Forward(input_vec);
	double time1 = cvGetTickCount();

	float time_cost = (time1 - time0) / cv::getTickFrequency() * 1000;
	std::cout << "Total cost time : " << time_cost << std::endl;
	int		n_outCount = output_vec[0]->count();
	int		n_num = output_vec[0]->num();
	int		n_width = output_vec[0]->width();
	int		n_height = output_vec[0]->height();
	int		n_channels = output_vec[0]->channels();
	float	*output_maps = new float[n_num * n_width * n_height * n_channels];
	for (int i = 0; i < output_vec.size(); i++) {
		memcpy(output_maps, output_vec[i]->cpu_data(), sizeof(float)*n_outCount);

		//for (int c = 0; c < n_channels; ++c) {	
		for (int c = 0; c < max_keypoints; ++c) {
			cv::Mat heatmap = cv::Mat::zeros(cv::Size(n_width, n_height), CV_32FC1);
			float *iptr = (float*)heatmap.data;
			for (int h = 0; h < n_height; ++h) {
				for (int w = 0; w < n_width; ++w) {
					iptr[h * n_width + w] = output_maps[c * n_height * n_width + h * n_width + w];
				}
			}
			cv::Mat resize_heatmap;
			cv::resize(heatmap, resize_heatmap, { resize_width_keypoints, resize_height_keypoints }, 0, 0, CV_INTER_CUBIC);
			//cv::imshow("heatmap", resize_heatmap);
			//cv::waitKey(0);
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			// 基于opencv寻找HeatMap中的最大最小值
			cv::minMaxLoc(resize_heatmap, &minVal, &maxVal, &minLoc, &maxLoc);
			cv::Point3f point;
			point.x = float(maxLoc.x);
			point.y = float(maxLoc.y);
			point.z = float(maxVal);
			face_keypoints.push_back(resizeSrcPoint3f(point, resize_width_keypoints, resize_height_keypoints, src_image.cols, src_image.rows));
		}
	}

	delete[] output_maps;
	for (int i = 0; i < input_vec.size(); ++i) {
		if (input_vec[i] != NULL) {
			delete input_vec[i];
		}
	}
	return face_keypoints;
}
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// one heatmap
vector<cv::Point3f> predictFaceKeypointsOneHeatMap(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_keypoints, int resize_height_keypoints, int max_keypoints = KEYPOINTS_NUMBER) {
	vector<cv::Point3f> face_keypoints;
	cv::Mat image = createResizeImage(src_image, resize_width_keypoints, resize_height_keypoints);
	vector<Blob<float>*>	input_vec;
	vector<Blob<float>*>	output_vec;
	if (!getInputVec(image, input_vec)) {
		return face_keypoints;
	}

	double time0 = cvGetTickCount();
	output_vec = caffe_net->Forward(input_vec);
	double time1 = cvGetTickCount();

	float time_cost = (time1 - time0) / cv::getTickFrequency() * 1000;
	std::cout << "Total cost time : " << time_cost << std::endl;
	int		n_outCount = output_vec[0]->count();
	int		n_num = output_vec[0]->num();
	int		n_width = output_vec[0]->width();
	int		n_height = output_vec[0]->height();
	int		n_channels = output_vec[0]->channels();
	float	*output_maps = new float[n_outCount];
	for (int i = 0; i < output_vec.size(); i++) {
		memcpy(output_maps, output_vec[i]->cpu_data(), sizeof(float)*n_outCount);
		for (int j = 0; j < n_outCount; j = j + 3) {
			cv::Point3f point;
			point.x = output_maps[j];
			point.y = output_maps[j+1];
			point.z = output_maps[j+2];
			point.x = point.x * resize_width_keypoints / 2.0 + resize_width_keypoints / 2.0;
			point.y = point.y * resize_height_keypoints / 2.0 + resize_height_keypoints / 2.0;
			face_keypoints.push_back(resizeSrcPoint3f(point, resize_width_keypoints, resize_height_keypoints, src_image.cols, src_image.rows));
		}
	}

	delete[] output_maps;
	for (int i = 0; i < input_vec.size(); ++i) {
		if (input_vec[i] != NULL) {
			delete input_vec[i];
		}
	}
	return face_keypoints;
}
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

void number2str(const float &int_temp, string &string_temp)
{
	stringstream stream;
	stream << int_temp;
	string_temp = stream.str();   //此处也可以用 stream>>string_temp  
}

// time cost all layers
void timeCost(Net<float> *caffe_net, const int iterators) {
	const vector<boost::shared_ptr<Layer<float> > >& layers = caffe_net->layers();
	const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net->bottom_vecs();
	const vector<vector<Blob<float>*> >& top_vecs = caffe_net->top_vecs();
	const vector<vector<bool> >& bottom_need_backward =
		caffe_net->bottom_need_backward();

	LOG(INFO) << "*** Benchmark begins ***";
	LOG(INFO) << "Testing for " << iterators << " iterations.";
	Timer total_timer;
	total_timer.Start();
	Timer forward_timer;
	Timer backward_timer;
	Timer timer;
	std::vector<double> forward_time_per_layer(layers.size(), 0.0);
	std::vector<double> backward_time_per_layer(layers.size(), 0.0);
	double forward_time = 0.0;
	double backward_time = 0.0;
	for (int j = 0; j < iterators; ++j) {
		Timer iter_timer;
		iter_timer.Start();
		forward_timer.Start();
		for (int i = 0; i < layers.size(); ++i) {
			timer.Start();
			layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
			forward_time_per_layer[i] += timer.MicroSeconds();
		}
		forward_time += forward_timer.MicroSeconds();
		LOG(INFO) << "Iteration: " << j + 1 << " forward time: "
			<< iter_timer.MilliSeconds() << " ms.";
	}
	LOG(INFO) << "Average time per layer: ";
	for (int i = 0; i < layers.size(); ++i) {
		const caffe::string& layername = layers[i]->layer_param().name();
		LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
			"\tforward: " << forward_time_per_layer[i] / 1000 /
			iterators << " ms.";
	}
	total_timer.Stop();
	LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
		iterators << " ms.";
	LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
	LOG(INFO) << "*** Benchmark ends ***";
}

// ######################################################################################
void showMatInformation(cv::Mat img) {
	//cout << img << endl;

	cout << "dims:" << img.dims << endl;
	cout << "rows:" << img.rows << endl;
	cout << "cols:" << img.cols << endl;
	cout << "channels:" << img.channels() << endl;
	cout << "type:" << img.type() << endl;
	cout << "depth:" << img.depth() << endl;
	cout << "elemSize:" << img.elemSize() << endl;
	cout << "elemSize1:" << img.elemSize1() << endl;
	cv::Mat t_1(3, 4, CV_32FC3, Scalar_<float>(1, 2, 3));
	cv::Mat t_2(3, 4, CV_8UC3, Scalar_<float>(1, 2, 3));
	cv::Mat t_3(3, 4, CV_16SC3, Scalar_<float>(1, 2, 3));
	cout << "CV_32FC3 type:" << t_1.type() << endl;
	cout << "CV_16SC3 type:" << t_3.type() << endl;
	cout << "CV_8UC3 type:" << t_2.type() << endl;
}
// add clown nouse
uchar addTwoUchar(uchar a, uchar b, float ratio){
	if (b == 0) {
		return a;
	}
	else if (a == 0) {
		return b;
	}
	else {
		float temp = (1 - ratio) * a + ratio * b;
		if (temp > 255) {
			return 255;
		}
		else {
			return uchar(temp);
		}
	}
}

cv::Mat drawClownNouse(cv::Mat src_img, cv::Mat clown_nouse, cv::Point p_center, float ratio = 0.5) {
	cv::Mat draw_img = src_img.clone();
	//showMatInformation(draw_img);
	//showMatInformation(clown_nouse);
	int step_h = clown_nouse.rows / 2;
	int step_w = clown_nouse.cols / 2;
	int start_x = (p_center.x - step_w);
	int start_y = (p_center.y - step_h);
	int end_x = (p_center.x + step_w);
	int end_y = (p_center.y + step_h);
	start_x = start_x >= 0 ? start_x : 0;
	start_y = start_y >= 0 ? start_y : 0;
	end_x = end_x > src_img.cols ? src_img.cols - 1 : end_x;
	end_y = end_y > src_img.rows ? src_img.rows - 1 : end_y;
	for (int x = start_x; x <= end_x; ++x) {
		for (int y = start_y; y <= end_y; ++y) {
			if (draw_img.channels() == 3) {
				Vec3b old = src_img.at<Vec3b>(y, x);
				if (abs(clown_nouse.at<Vec3b>(y - start_y, x - start_x)[0] - clown_nouse.at<Vec3b>(y - start_y, x - start_x)[1]) < 10 && abs(clown_nouse.at<Vec3b>(y - start_y, x - start_x)[1] - clown_nouse.at<Vec3b>(y - start_y, x - start_x)[2]) < 10) {
					draw_img.at<Vec3b>(y, x) = old;
				}
				else {
					uchar a = addTwoUchar(old[0], clown_nouse.at<Vec3b>(y - start_y, x - start_x)[0], ratio);
					uchar b = addTwoUchar(old[1], clown_nouse.at<Vec3b>(y - start_y, x - start_x)[1], ratio);
					uchar c = addTwoUchar(old[2], clown_nouse.at<Vec3b>(y - start_y, x - start_x)[2], ratio);
					draw_img.at<Vec3b>(y, x) = Vec3b(a, b, c);
				}
			}
			else {
				//draw_img.at<float>(y, x) = (1 - ratio) * src_img.at<float>(y, x) + ratio * clown_nouse.at<float>(y - start_y, x - start_x);
			}
		}
	}
	return draw_img;
}
cv::Mat drawFace(cv::Mat image, vector<cv::Point3f>  face_keypoints, int left_src, int top_src, float keypoint_thred, cv::Mat cheek, cv::Mat nouse, cv::Mat left_ear_img, cv::Mat right_ear_img) {
	cv::Mat temp_img = image.clone();
	{
		// draw nouse
		cv::Point3f point = face_keypoints[30];
		if (point.z >= keypoint_thred) {
			temp_img = drawClownNouse(temp_img, nouse, cv::Point(point.x, point.y) + cv::Point(left_src, top_src), 1.0);
		}
	}

	{
		/*
		// draw cheek
		cv::Point3f point1 = face_keypoints[2];
		cv::Point3f point2 = face_keypoints[31];
		cv::Point3f point3 = face_keypoints[35];
		cv::Point3f point4 = face_keypoints[14];
		if (point1.z >= keypoint_thred && point2.z >= keypoint_thred && point3.z >= keypoint_thred && point4.z >= keypoint_thred) {
			cv::Point left_cheek = cv::Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2);
			cv::Point right_cheek = cv::Point((point3.x + point4.x) / 2, (point3.y + point4.y) / 2);
			temp_img = drawClownNouse(temp_img, cheek, left_cheek + cv::Point(left_src, top_src));
			temp_img = drawClownNouse(temp_img, cheek, right_cheek + cv::Point(left_src, top_src));
		}
		*/
	}

	{
		// draw ear
		/*
		cv::Point3f point1 = face_keypoints[17];
		cv::Point3f point2 = face_keypoints[31];
		cv::Point3f point3 = face_keypoints[26];
		cv::Point3f point4 = face_keypoints[35];

		if (point1.z >= keypoint_thred && point2.z >= keypoint_thred && point3.z >= keypoint_thred && point4.z >= keypoint_thred) {
			float ratio = 1.8;
			cv::Point left_ear = cv::Point(ratio * point1.x + (1 - ratio) * point2.x, ratio * point1.y + (1 - ratio) * point2.y);
			cv::Point right_ear = cv::Point(ratio * point3.x + (1 - ratio) * point4.x, ratio * point3.y + (1 - ratio) * point4.y);
			temp_img = drawClownNouse(temp_img, left_ear_img, left_ear + cv::Point(left_src, top_src), 1.0);
			temp_img = drawClownNouse(temp_img, right_ear_img, right_ear + cv::Point(left_src, top_src), 1.0);
		}
		*/
		cv::Point3f point1 = face_keypoints[1];
		cv::Point3f point2 = face_keypoints[15];

		if (point1.z >= keypoint_thred && point2.z >= keypoint_thred) {
			float ratio = 1.1;
			cv::Point left_ear = cv::Point(ratio * point1.x + (1 - ratio) * point2.x, ratio * point1.y + (1 - ratio) * point2.y);
			cv::Point right_ear = cv::Point(ratio * point2.x + (1 - ratio) * point1.x, ratio * point2.y + (1 - ratio) * point1.y);
			temp_img = drawClownNouse(temp_img, left_ear_img, left_ear + cv::Point(left_src, top_src), 1.0);
			temp_img = drawClownNouse(temp_img, right_ear_img, right_ear + cv::Point(left_src, top_src), 1.0);
		}
	}
	return temp_img;
}

// ######################################################################################
int main(int argc, char** argv) {
	//::google::InitGoogleLogging(argv[0]);
	//::google::ShutdownGoogleLogging();
	// shutdown log
	// 0 - debug
	// 1 - info(still a LOT of outputs)
	// 2 - warnings
	// 3 - errors
	//fLI::FLAGS_minloglevel = 2;
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif
	google::SetCommandLineOption("GLOG_minloglevel", "1");
	gflags::SetUsageMessage("face test\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    test_net [step] image_name\n"
	);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	/*
	if (argc < 2) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}
	*/
	int step = FLAGS_step;

	if (step == 1) {
		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir = FLAGS_model_dir_detection;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_detection;
		std::string model_params = model_dir + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		bool is_use_label_file = FLAGS_is_use_label_file_detection;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_detection = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		const bool is_color = !FLAGS_gray;

		// Test one image
		string image_file_name = argv[1];
		string label_file = "";
		//FaceInf face_information;
		if (is_use_label_file) {
			label_file = argv[2];
			//face_information = loadKeypointsFromFile(label_file);
		}
		cv::Mat src_image;
		if (is_color) {
			src_image = cv::imread(image_file_name);
		}
		else {
			src_image = cv::imread(image_file_name, CV_LOAD_IMAGE_GRAYSCALE);
		}
		std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

		cv::Mat temp_img = src_image.clone();
		bool flag = false;
		for (int kk = 0; kk < face_detection_all.size(); ++kk) {

			int left_src = face_detection_all[kk].x;
			int top_src = face_detection_all[kk].y;
			int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
			int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
			cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
			if (!flag) {
				cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
				cv::imwrite(output_one_detection, rect_img);
			}
		}
		cv::imshow("Detection Result", temp_img);
		cv::waitKey(0);
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 2) {
		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir = FLAGS_model_dir_keypoints;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_keypoints;
		std::string model_params = model_dir + FLAGS_model_params_keypoints;
		bool is_use_gpu = FLAGS_is_use_gpu;
		bool is_use_label_file = FLAGS_is_use_label_file_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		const bool is_color = !FLAGS_gray;

		// Test one image
		string image_file_name = argv[1];
		string label_file = "";
		//FaceInf face_information;
		if (is_use_label_file) {
			label_file = argv[2];
			//face_information = loadKeypointsFromFile(label_file);
		}
		cv::Mat src_image;
		if (is_color) {
			src_image = cv::imread(image_file_name);
		}
		else {
			src_image = cv::imread(image_file_name, CV_LOAD_IMAGE_GRAYSCALE);
		}
		vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe_net_keypoints, src_image, resize_width_keypoints, resize_height_keypoints);
		cv::Mat temp_img = src_image.clone();
		bool flag = false;
		for (int kk = 0; kk < face_keypoints.size(); ++kk) {
			cv::Point3f point = face_keypoints[kk];
			if (point.z < keypoint_thred) {
				continue;
			}
			std::cout << point.z << std::endl;
			/*
			string prob_str = "";
			number2str(point.z, prob_str);
			cv::putText(temp_img, prob_str, cv::Point(point.x - 5, point.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));

			string index_str = "";
			number2str(kk+1, index_str);
			cv::putText(temp_img, index_str, cv::Point(point.x + 5, point.y + 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));
			*/
			cv::circle(temp_img, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), -1);

		}
		cv::imshow("Keypoints Result", temp_img);
		cv::waitKey(0);
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 3) {
		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir_detection = FLAGS_model_dir_detection;
		std::string model_prototxt_detection = model_dir_detection + FLAGS_model_prototxt_detection;
		std::string model_params_detection = model_dir_detection + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, is_use_gpu);

		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir_keypoints = FLAGS_model_dir_keypoints;
		std::string model_prototxt_keypoints = model_dir_keypoints + FLAGS_model_prototxt_keypoints;
		std::string model_params_keypoints = model_dir_keypoints + FLAGS_model_params_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt_keypoints, model_params_keypoints, is_use_gpu);

		const bool is_color = !FLAGS_gray;

		// Test one image
		string image_file_name = argv[1];

		cv::Mat src_image;
		if (is_color) {
			src_image = cv::imread(image_file_name);
		}
		else {
			src_image = cv::imread(image_file_name, CV_LOAD_IMAGE_GRAYSCALE);
		}
		std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

		cv::Mat temp_img = src_image.clone();
		for (int kk = 0; kk < face_detection_all.size(); ++kk) {
			int left_src = face_detection_all[kk].x;
			int top_src = face_detection_all[kk].y;
			int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
			int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
			cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
			cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
			if (rect_img.cols * rect_img.rows < 40 * 40) {
				continue;
			}
			vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe_net_keypoints, rect_img, resize_width_keypoints, resize_height_keypoints);
			for (int tt = 0; tt < face_keypoints.size(); ++tt) {
				cv::Point3f point = face_keypoints[tt];
				if (point.z < keypoint_thred) {
					continue;
				}
				cv::circle(temp_img, cv::Point(point.x, point.y) + cv::Point(left_src, top_src), 2, cv::Scalar(0, 0, 255), -1);
			}
		}
		cv::imshow("Full Result", temp_img);
		cv::waitKey(0);
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 4) {
		
		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir_detection = FLAGS_model_dir_detection;
		std::string model_prototxt_detection = model_dir_detection + FLAGS_model_prototxt_detection;
		std::string model_params_detection = model_dir_detection + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		//caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, is_use_gpu);
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, false);

		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir_keypoints = FLAGS_model_dir_keypoints;
		std::string model_prototxt_keypoints = model_dir_keypoints + FLAGS_model_prototxt_keypoints;
		std::string model_params_keypoints = model_dir_keypoints + FLAGS_model_params_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt_keypoints, model_params_keypoints, is_use_gpu);
		const bool is_color = !FLAGS_gray;

		VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
		for (;;)
		{
			Mat src_image;
			cap >> src_image; // get a new frame from camera
			Caffe::set_mode(Caffe::CPU);
			std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

			std::cout << "Detection number: " << face_detection_all.size() << std::endl;
			cv::Mat temp_img = src_image.clone();
			for (int kk = 0; kk < face_detection_all.size(); ++kk) {
				int left_src = face_detection_all[kk].x;
				int top_src = face_detection_all[kk].y;
				int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
				int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
				cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
				cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
				if (rect_img.cols * rect_img.rows < 40 * 40) {
					continue;
				}
				Caffe::set_mode(Caffe::GPU);
				vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe_net_keypoints, rect_img, resize_width_keypoints, resize_height_keypoints);
				for (int tt = 0; tt < face_keypoints.size(); ++tt) {
					cv::Point3f point = face_keypoints[tt];
					if (point.z < keypoint_thred) {
						continue;
					}
					cv::circle(temp_img, cv::Point(point.x, point.y) + cv::Point(left_src, top_src), 2, cv::Scalar(0, 0, 255), -1);
				}
			}
			cv::imshow("Full Result", temp_img);
			char ch = cv::waitKey(1);
			if (ch == 'q') {
				break;
			}
		}
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}
		
	}
	else if (step == 5) {
		// only face keypoints.
		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir = FLAGS_model_dir_keypoints;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_keypoints;
		std::string model_params = model_dir + FLAGS_model_params_keypoints;
		bool is_use_gpu = FLAGS_is_use_gpu;
		bool is_use_label_file = FLAGS_is_use_label_file_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		const bool is_color = !FLAGS_gray;

		// Test one image
		string image_file_name = argv[1];
		string label_file = "";
		//FaceInf face_information;
		if (is_use_label_file) {
			label_file = argv[2];
			//face_information = loadKeypointsFromFile(label_file);
		}
		cv::Mat src_image;
		if (is_color) {
			src_image = cv::imread(image_file_name);
		}
		else {
			src_image = cv::imread(image_file_name, CV_LOAD_IMAGE_GRAYSCALE);
		}
		vector<cv::Point3f> face_keypoints = predictFaceKeypointsOneHeatMap(caffe_net_keypoints, src_image, resize_width_keypoints, resize_height_keypoints);
		cv::Mat temp_img = src_image.clone();
		bool flag = false;
		for (int kk = 0; kk < face_keypoints.size(); ++kk) {
			cv::Point3f point = face_keypoints[kk];
			if (point.z < keypoint_thred) {
				continue;
			}
			/*
			string prob_str = "";
			number2str(point.z, prob_str);
			cv::putText(temp_img, prob_str, cv::Point(point.x - 5, point.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));

			string index_str = "";
			number2str(kk+1, index_str);
			cv::putText(temp_img, index_str, cv::Point(point.x + 5, point.y + 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));
			*/
			cv::circle(temp_img, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), -1);

		}
		cv::imshow("Keypoints Result", temp_img);
		cv::waitKey(0);
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 6) {
		// one heatmap for 68 points
		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir_detection = FLAGS_model_dir_detection;
		std::string model_prototxt_detection = model_dir_detection + FLAGS_model_prototxt_detection;
		std::string model_params_detection = model_dir_detection + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, is_use_gpu);

		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir_keypoints = FLAGS_model_dir_keypoints;
		std::string model_prototxt_keypoints = model_dir_keypoints + FLAGS_model_prototxt_keypoints;
		std::string model_params_keypoints = model_dir_keypoints + FLAGS_model_params_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt_keypoints, model_params_keypoints, is_use_gpu);

		const bool is_color = !FLAGS_gray;

		VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
		for (;;)
		{
			Mat src_image;
			cap >> src_image; // get a new frame from camera

			std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

			cv::Mat temp_img = src_image.clone();
			for (int kk = 0; kk < face_detection_all.size(); ++kk) {
				int left_src = face_detection_all[kk].x;
				int top_src = face_detection_all[kk].y;
				int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
				int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
				cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
				cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
				if (rect_img.cols * rect_img.rows < 40 * 40) {
					continue;
				}
				vector<cv::Point3f> face_keypoints = predictFaceKeypointsOneHeatMap(caffe_net_keypoints, src_image, resize_width_keypoints, resize_height_keypoints);
				for (int tt = 0; tt < face_keypoints.size(); ++tt) {
					cv::Point3f point = face_keypoints[tt];
					if (point.z < keypoint_thred) {
						continue;
					}
					cv::circle(temp_img, cv::Point(point.x, point.y) + cv::Point(left_src, top_src), 2, cv::Scalar(0, 0, 255), -1);
				}
			}
			cv::imshow("Full Result", temp_img);
			char ch = cv::waitKey(1);
			if (ch == 'q') {
				break;
			}
		}
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 7) {
		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir = FLAGS_model_dir_keypoints;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_keypoints;
		std::string model_params = model_dir + FLAGS_model_params_keypoints;
		bool is_use_gpu = FLAGS_is_use_gpu;
		bool is_use_label_file = FLAGS_is_use_label_file_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		timeCost(caffe_net, FLAGS_iterations);
		system("pause");
	}
	else if (step == 8) {
		string cheek_img = "F:/project/face/face_api/caffe-windows/models/clown_nouse.png";
		cv::Mat cheek = cv::imread(cheek_img);
		cv::Mat cheek_resize = createResizeImage(cheek, 40, 40);

		string nouse_img = "F:/project/face/face_api/caffe-windows/models/nouse.png";
		cv::Mat nouse = cv::imread(nouse_img);
		cv::Mat nouse_resize = createResizeImage(nouse, 60, 60);

		string left_ear_img = "F:/project/face/face_api/caffe-windows/models/left_ear.png";
		cv::Mat left_ear = cv::imread(left_ear_img);
		cv::Mat left_ear_resize = createResizeImage(left_ear, 60, 60);

		string right_ear_img = "F:/project/face/face_api/caffe-windows/models/right_ear.png";
		cv::Mat right_ear = cv::imread(right_ear_img);
		cv::Mat right_ear_resize = createResizeImage(right_ear, 60, 60);

		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir_detection = FLAGS_model_dir_detection;
		std::string model_prototxt_detection = model_dir_detection + FLAGS_model_prototxt_detection;
		std::string model_params_detection = model_dir_detection + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, is_use_gpu);

		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir_keypoints = FLAGS_model_dir_keypoints;
		std::string model_prototxt_keypoints = model_dir_keypoints + FLAGS_model_prototxt_keypoints;
		std::string model_params_keypoints = model_dir_keypoints + FLAGS_model_params_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt_keypoints, model_params_keypoints, is_use_gpu);

		const bool is_color = !FLAGS_gray;

		VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
		for (;;)
		{
			Mat src_image;
			cap >> src_image; // get a new frame from camera

			std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

			cv::Mat temp_img = src_image.clone();
			for (int kk = 0; kk < face_detection_all.size(); ++kk) {
				int left_src = face_detection_all[kk].x;
				int top_src = face_detection_all[kk].y;
				int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
				int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
				cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
				cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
				if (rect_img.cols * rect_img.rows < 40 * 40) {
					continue;
				}
				vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe_net_keypoints, rect_img, resize_width_keypoints, resize_height_keypoints);
				temp_img = drawFace(temp_img, face_keypoints, left_src, top_src, keypoint_thred, cheek_resize, nouse_resize, left_ear_resize, right_ear_resize);

			}
			cv::imshow("Full Result", temp_img);
			char ch = cv::waitKey(1);
			if (ch == 'q') {
				break;
			}
		}
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}

	}
	else if (step == 9) {
		int resize_height_keypoints = std::max<int>(0, FLAGS_resize_height_keypoints);
		int resize_width_keypoints = std::max<int>(0, FLAGS_resize_width_keypoints);
		std::string model_dir = FLAGS_model_dir_keypoints;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_keypoints;
		std::string model_params = model_dir + FLAGS_model_params_keypoints;
		bool is_use_gpu = FLAGS_is_use_gpu;
		bool is_use_label_file = FLAGS_is_use_label_file_keypoints;
		std::string output_one_keypoints = FLAGS_output_name_keypoints;
		float keypoint_thred = FLAGS_keypoint_thred;
		// init the caffe net
		Net<float> *caffe_net_keypoints = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		const bool is_color = !FLAGS_gray;

		VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
		for (;;)
		{
			Mat src_image;
			cap >> src_image; // get a new frame from camera

			vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe_net_keypoints, src_image, resize_width_keypoints, resize_height_keypoints);
			cv::Mat temp_img = src_image.clone();
			bool flag = false;
			for (int kk = 0; kk < face_keypoints.size(); ++kk) {
				cv::Point3f point = face_keypoints[kk];
				if (point.z < keypoint_thred) {
					continue;
				}
				/*
				string prob_str = "";
				number2str(point.z, prob_str);
				cv::putText(temp_img, prob_str, cv::Point(point.x - 5, point.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));

				string index_str = "";
				number2str(kk+1, index_str);
				cv::putText(temp_img, index_str, cv::Point(point.x + 5, point.y + 5), cv::FONT_HERSHEY_COMPLEX, 0.25, Scalar(0, 255, 255));
				*/
				cv::circle(temp_img, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), -1);

			}
			cv::imshow("Keypoints Result", temp_img);
			char ch = cv::waitKey(1);
			if (ch == 'q') {
				break;
			}
		}
		if (caffe_net_keypoints != NULL) {
			delete caffe_net_keypoints;
			caffe_net_keypoints = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	else if (step == 10) {
		// --step=10 -is_use_gpu=true  F:\project\face\face_api\clcaffe-windows\caffe\build\tools\test\test.bmp
		std::string model_dir = FLAGS_model_dir_detection;
		std::string model_prototxt = model_dir + FLAGS_model_prototxt_detection;
		std::string model_params = model_dir + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		// init the caffe net
		Net<float> *caffe_net = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net = initCaffeNet(model_prototxt, model_params, is_use_gpu);
		timeCost(caffe_net, FLAGS_iterations);
		system("pause");
	}
	else if (step == 11){

		int max_people = std::max<int>(1, FLAGS_max_people);
		int resize_height_detection = std::max<int>(0, FLAGS_resize_height_detection);
		int resize_width_detection = std::max<int>(0, FLAGS_resize_width_detection);
		std::string model_dir_detection = FLAGS_model_dir_detection;
		std::string model_prototxt_detection = model_dir_detection + FLAGS_model_prototxt_detection;
		std::string model_params_detection = model_dir_detection + FLAGS_model_params_detection;
		bool is_use_gpu = FLAGS_is_use_gpu;
		std::string output_one_detection = FLAGS_output_name_detection;
		// init the caffe net
		Net<float> *caffe_net_detection = NULL;
		LOG(INFO) << "Loading net ...";
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, is_use_gpu);

		const bool is_color = !FLAGS_gray;

		VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
		for (;;)
		{
			Mat src_image;
			cap >> src_image; // get a new frame from camera
			//Caffe::set_mode(Caffe::CPU);
			std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_net_detection, src_image, resize_width_detection, resize_height_detection, max_people);

			std::cout << "Detection number: " << face_detection_all.size() << std::endl;
			cv::Mat temp_img = src_image.clone();
			for (int kk = 0; kk < face_detection_all.size(); ++kk) {
				int left_src = face_detection_all[kk].x;
				int top_src = face_detection_all[kk].y;
				int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
				int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
				cv::rectangle(temp_img, cv::Point(left_src, top_src), cv::Point(right_src, bot_src), cv::Scalar(0, 0, 255), 2);
			}
			cv::imshow("Full Result", temp_img);
			char ch = cv::waitKey(1);
			if (ch == 'q') {
				break;
			}
		}
		if (caffe_net_detection != NULL) {
			delete caffe_net_detection;
			caffe_net_detection = NULL;
			LOG(INFO) << "Release net success.";
		}
	}
	return 0;
}
