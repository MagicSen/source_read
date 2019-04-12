#include "face_sdk_implement.h"

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

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

namespace face_sdk {
	const int g_max_people = 10;
	const int g_keypoints_number = 68;
	// detection params
	const int g_resize_height_detection = 120;
	const int g_resize_width_detection = 160;
	//const std::string g_model_prototxt_detection = "detection_model.prototxt";
	//const std::string g_model_params_detection = "detection_model.caffemodel.h5";
	const std::string g_model_prototxt_detection = "a.prototxt";
	const std::string g_model_params_detection = "a.h5";

	// keypoints params
	const int g_resize_height_keypoints = 240;
	const int g_resize_width_keypoints = 240;
	//const std::string g_model_prototxt_keypoints = "keypoints_model.prototxt";
	//const std::string g_model_params_keypoints = "keypoints_model.caffemodel.h5";
	const std::string g_model_prototxt_keypoints = "b.prototxt";
	const std::string g_model_params_keypoints = "b.h5";
	const float g_keypoint_thred = 0.4;

	//const bool g_is_use_gpu = false;
	const bool g_is_use_gpu = true;
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

	cv::Mat getImageFromPoints(float *img_data, int src_width, int src_height, bool is_color);
	Net<float>* initCaffeNet(std::string caffe_net_prototxt_file, std::string caffe_net_params_file, bool is_use_gpu);
	void deleteVoidPoints(void *p);
	bool getInputVec(cv::Mat src_image, vector<Blob<float> *> &input_vec);
	// detection api
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
	Rectangle resizeNormaRectangle(float x, float y, float w, float h, int width, int height, int type = 1, float ratio = 1.4);
	Rectangle resizeSrcRectangle(Rectangle resize_rectangle, int resize_width, int resize_height, int img_width, int img_height);
	cv::Mat createResizeImage(cv::Mat img, const int &resize_width, const int &resize_height);
	std::vector<Rectangle> predictFaceDetection(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_detection, int resize_height_detection, int max_people = g_max_people);

	// keypoints api
	Point3f resizeSrcPoint3f(Point3f point, int resize_width, int resize_height, int img_width, int img_height);
	vector<cv::Point3f> predictFaceKeypoints(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_keypoints, int resize_height_keypoints, int max_keypoints = g_keypoints_number);


	FaceSDKImplement::FaceSDKImplement()
	{
	}

	FaceSDKImplement::FaceSDKImplement(std::string model_path)
	{
		p_modelPath = model_path;
		caffe_net_detection = NULL;
		caffe_net_keypoints = NULL;
		p_isAvailable = init();
	}

	FaceSDKImplement::~FaceSDKImplement()
	{
		// auto delete
		//deleteVoidPoints(caffe_net_detection);
		//deleteVoidPoints(caffe_net_keypoints);
	}

	bool FaceSDKImplement::init()
	{
		// shutdown log
		// 0 - debug
		// 1 - info(still a LOT of outputs)
		// 2 - warnings
		// 3 - error
		fLI::FLAGS_minloglevel = 2;
		deleteVoidPoints(caffe_net_detection);
		deleteVoidPoints(caffe_net_keypoints);

		std::string model_prototxt_detection = p_modelPath + g_model_prototxt_detection;
		std::string model_params_detection = p_modelPath + g_model_params_detection;
		LOG(INFO) << "Loading detection net ...";
		//caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, g_is_use_gpu);
		caffe_net_detection = initCaffeNet(model_prototxt_detection, model_params_detection, false);

		std::string model_prototxt_keypoints = p_modelPath + g_model_prototxt_keypoints;
		std::string model_params_keypoints = p_modelPath + g_model_params_keypoints;
		LOG(INFO) << "Loading keypoints net ...";
		caffe_net_keypoints = initCaffeNet(model_prototxt_keypoints, model_params_keypoints, g_is_use_gpu);

		if (caffe_net_detection == NULL || caffe_net_keypoints == NULL) {
			return false;
		}
		else {
			return true;
		}
	}

	bool FaceSDKImplement::isAvailable()
	{
		return p_isAvailable;
	}

	std::vector<Person> FaceSDKImplement::predict(float *img_data, int src_width, int src_height, bool is_color)
	{
		std::vector<Person> predict_result;
		if (!isAvailable() || img_data == NULL) {
			return predict_result;
		}

		cv::Mat src_image = getImageFromPoints(img_data, src_width, src_height, is_color);
		// debug
		//cv::imshow("Src image", src_image);
		//cv::waitKey(0);
		Net<float> *caffe_detection = (Net<float> *)caffe_net_detection;
		Net<float> *caffe__keypoints = (Net<float> *)caffe_net_keypoints;
		Caffe::set_mode(Caffe::CPU);
		std::vector<Rectangle> face_detection_all = predictFaceDetection(caffe_detection, src_image, g_resize_width_detection, g_resize_height_detection, g_max_people);
		Caffe::set_mode(Caffe::GPU);
		cv::Mat temp_img = src_image.clone();
		for (int kk = 0; kk < face_detection_all.size(); ++kk) {
			int left_src = face_detection_all[kk].x;
			int top_src = face_detection_all[kk].y;
			int right_src = face_detection_all[kk].x + face_detection_all[kk].width;
			int bot_src = face_detection_all[kk].y + face_detection_all[kk].height;
			cv::Mat rect_img = src_image(cv::Rect(face_detection_all[kk].x, face_detection_all[kk].y, face_detection_all[kk].width, face_detection_all[kk].height));
			if (rect_img.cols * rect_img.rows < 40 * 40) {
				continue;
			}
			vector<cv::Point3f> face_keypoints = predictFaceKeypoints(caffe__keypoints, rect_img, g_resize_width_keypoints, g_resize_height_keypoints);
			Person p;
			p.p_rectangle.x = face_detection_all[kk].x;
			p.p_rectangle.y = face_detection_all[kk].y;
			p.p_rectangle.w = face_detection_all[kk].width;
			p.p_rectangle.h = face_detection_all[kk].height;
			for (int i = 0; i < face_keypoints.size(); ++i) {
				PKeyPoint p_keypoint;
				p_keypoint.x = face_keypoints[i].x;
				p_keypoint.y = face_keypoints[i].y;
				p_keypoint.p = face_keypoints[i].z;
				p.p_keypoints.push_back(p_keypoint);
			}
			predict_result.push_back(p);
		}
		return predict_result;
	}

	cv::Mat getImageFromPoints(float *img_data, int src_width, int src_height, bool is_color) {
		cv::Mat img = cv::Mat::zeros(Size(src_width, src_height), CV_8UC3);
		int channels = 3;
		if (!is_color) {
			channels = 1;
		}
		uchar *iptr = (uchar*)img.data;

		for (int w = 0; w < src_width; ++w) {
			for (int h = 0; h < src_height; ++h) {
				for (int c = 0; c < channels; ++c) {
					iptr[h * src_width * channels + w * channels + c] = img_data[c * src_height * src_width + h * src_width + w];
				}
			}
		}
		return img;
	}

	Net<float>* initCaffeNet(std::string caffe_net_prototxt_file, std::string caffe_net_params_file, bool is_use_gpu) {
		/*
		vector<int> gpus;
		get_gpus(&gpus);
		if (gpus.size() == 0 || !is_use_gpu) {
			//std::cout << "Using CPU" << std::endl;
			Caffe::set_mode(Caffe::CPU);
			NetParameter net_prototxt;

			ReadNetParamsFromTextFileOrDie(caffe_net_prototxt_file, &net_prototxt);
			Net<float> *caffe_net = new Net<float>(net_prototxt, Caffe::GetDefaultDevice());
			caffe_net->CopyTrainedLayersFromHDF5(caffe_net_params_file);
			return caffe_net;
		}
		else {
			//std::cout << "Use GPU with device ID " << gpus[0] << std::endl;
			//Caffe::SetDevices(gpus);
			Caffe::set_mode(Caffe::GPU);
			Caffe::SetDevice(gpus[0]);

			Net<float> *caffe_net = new Net<float>(caffe_net_prototxt_file, caffe::Phase::TEST, Caffe::GetDefaultDevice());

			caffe_net->CopyTrainedLayersFromHDF5(caffe_net_params_file);
			return caffe_net;;
		}
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
			Caffe::set_mode(Caffe::GPU);
			Caffe::SetDevice(gpus[0]);

			Net<float> *caffe_net = new Net<float>(caffe_net_prototxt_file, caffe::Phase::TEST, Caffe::GetDefaultDevice());

			caffe_net->CopyTrainedLayersFromHDF5(caffe_net_params_file);
			return caffe_net;
		}
	}

	void deleteVoidPoints(void *p) {
		if (p != NULL) {
			delete p;
			p = NULL;
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


	Rectangle resizeNormaRectangle(float x, float y, float w, float h, int width, int height, int type, float ratio) {
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

	std::vector<Rectangle> predictFaceDetection(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_detection, int resize_height_detection, int max_people) {
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

	// #######################################################

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

	vector<cv::Point3f> predictFaceKeypoints(Net<float> *&caffe_net, cv::Mat src_image, int resize_width_keypoints, int resize_height_keypoints, int max_keypoints) {
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

	// #######################################################################


}
