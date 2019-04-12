#include <iostream>
#include "face_sdk_interface.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

// draw api
uchar addTwoUchar(uchar a, uchar b, float ratio) {
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
cv::Mat drawImageToOneImage(cv::Mat src_img, cv::Mat clown_nouse, face_sdk::PKeyPoint p_center, float ratio = 0.5) {
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
	for (int x = start_x; x < end_x; ++x) {
		for (int y = start_y; y < end_y; ++y) {
			if (draw_img.channels() == 3) {
				cv::Vec3b old = src_img.at<cv::Vec3b>(y, x);
				if (abs(clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[0] - clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[1]) < 10 && abs(clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[1] - clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[2]) < 10) {
					draw_img.at<cv::Vec3b>(y, x) = old;
				}
				else {
					uchar a = addTwoUchar(old[0], clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[0], ratio);
					uchar b = addTwoUchar(old[1], clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[1], ratio);
					uchar c = addTwoUchar(old[2], clown_nouse.at<cv::Vec3b>(y - start_y, x - start_x)[2], ratio);
					draw_img.at<cv::Vec3b>(y, x) = cv::Vec3b(a, b, c);
				}
			}
			else {
				// unsupport gray image
				// draw_img.at<uchar>(y, x) = (1 - ratio) * src_img.at<float>(y, x) + ratio * clown_nouse.at<float>(y - start_y, x - start_x);
			}
		}
	}
	return draw_img;
}

cv::Mat createResizeImage(cv::Mat img, const int &resize_width, const int &resize_height) {
	int img_height = img.rows;
	int img_width = img.cols;
	float ratio_width = float(resize_width) / img_width;
	float ratio_height = float(resize_height) / img_height;
	cv::Mat result = cv::Mat::zeros(cv::Size(resize_width, resize_height), img.type());
	if (ratio_width == ratio_height) {
		cv::resize(img, result, cv::Size(resize_width, resize_height));
	}
	else if (ratio_width < ratio_height) {
		// resize width first
		int new_resize_height = int(float(resize_width) / img_width * img_height);
		cv::Mat temp = cv::Mat::zeros(cv::Size(resize_width, new_resize_height), img.type());
		cv::resize(img, temp, cv::Size(resize_width, new_resize_height));
		int bias_h = (resize_height - new_resize_height) / 2.0;
		for (int h = 0; h < new_resize_height; ++h) {
			for (int w = 0; w < resize_width; ++w) {
				if (img.channels() == 3) {
					result.at<cv::Vec3b>(h + bias_h, w) = temp.at<cv::Vec3b>(h, w);
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
		cv::Mat temp = cv::Mat::zeros(cv::Size(new_resize_width, resize_height), img.type());
		cv::resize(img, temp, cv::Size(new_resize_width, resize_height));
		int bias_w = (resize_width - new_resize_width) / 2.0;
		for (int h = 0; h < resize_height; ++h) {
			for (int w = 0; w < new_resize_width; ++w) {
				if (img.channels() == 3) {
					result.at<cv::Vec3b>(h, w + bias_w) = temp.at<cv::Vec3b>(h, w);
				}
				else {
					result.at<uchar>(h, w + bias_w) = temp.at<uchar>(h, w);
				}
			}
		}

	}
	return result;
}
cv::Mat getNouseFromFile(std::string nouse_file, int width = 60, int height = 60) {
	cv::Mat nouse = cv::imread(nouse_file);
	cv::Mat nouse_resize = createResizeImage(nouse, width, height);
	return nouse_resize;
}

float *getMatToFloatPoint(cv::Mat img) {
	int height = img.rows;
	int width = img.cols;
	int channels = img.channels();
	float *img_data = new float[height * width * channels];
	uchar *iptr = (uchar*)img.data;
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			for (int c = 0; c < channels; ++c) {
				img_data[c * height * width + h * width + w] = iptr[h * width * channels + w * channels + c];
			}
		}
	}
	return img_data;
}

void test_video(std::string model_path){
	face_sdk::FaceSDKBase face_sdk_base(model_path);
	cv::VideoCapture cap(0); // open the default camera
	while (1){
		cv::Mat src_image;
		cap >> src_image; // get a new frame from camera
		float *img_data = getMatToFloatPoint(src_image);
		int height = src_image.rows;
		int width = src_image.cols;
		bool is_color = true;
		face_sdk_base.predict(img_data, width, height, is_color);
		// delete new bytes
		if (img_data != NULL) {
			delete[] img_data;
			img_data = NULL;
		}
		cv::Mat temp_img = src_image.clone();

		int number_person = face_sdk_base.getNumberPeople();
		for (int i = 0; i < number_person; ++i){
			face_sdk::Person person = face_sdk_base.getPersonKeypoints(i);
			if (!person.isAvailable()){
				continue;
			}
			person.p_rectangle.x;
			for (int j = 0; j < person.p_keypoints.size(); ++j){
				if (person.p_keypoints[j].p < 0.3){
					continue;
				}
				cv::circle(temp_img, cv::Point(person.p_keypoints[j].x, person.p_keypoints[j].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y), 2, cv::Scalar(0, 0, 255), -1);

			}
		}
		cv::imshow("Full Result", temp_img);
		char ch = cv::waitKey(1);
		if (ch == 'q') {
			break;
		}
	}
}

void test_singleimage(std::string model_path){
	face_sdk::FaceSDKBase face_sdk_base(model_path);

	std::string test_img = "../test/test.bmp";
	cv::Mat img = cv::imread(test_img);
	float *img_data = getMatToFloatPoint(img);
	int height = img.rows;
	int width = img.cols;
	bool is_color = true;
	// step 1 predict
	face_sdk_base.predict(img_data, width, height, is_color);
	// delete new bytes
	if (img_data != NULL) {
		delete[] img_data;
		img_data = NULL;
	}
	// step 2 get nouse or other position
	face_sdk::Nouse nouse = face_sdk_base.getNouse(0);
	if (nouse.available) {
		// step 3 draw the nouse
		std::string nouse_file = "../test/nouse.png";
		cv::Mat nouse_img = getNouseFromFile(nouse_file);
		cv::Mat draw_img = drawImageToOneImage(img, nouse_img, nouse.nouse_pos, 1);
		img = draw_img;
	}
	// show the image
	cv::imshow("Result", img);
	cv::waitKey(0);
}

void test_clear_img(std::string model_path, std::string image_name){
	face_sdk::FaceSDKBase face_sdk_base(model_path);
	cv::Mat src_image = cv::imread(image_name);
	float *img_data = getMatToFloatPoint(src_image);
	int height = src_image.rows;
	int width = src_image.cols;
	bool is_color = true;
	face_sdk_base.predict(img_data, width, height, is_color);
	// delete new bytes
	if (img_data != NULL) {
		delete[] img_data;
		img_data = NULL;
	}
	cv::Mat temp_img = src_image.clone();

	int number_person = face_sdk_base.getNumberPeople();
	for (int i = 0; i < number_person; ++i){
		face_sdk::Person person = face_sdk_base.getPersonKeypoints(i);
		if (!person.isAvailable()){
			continue;
		}
		person.p_rectangle.x;
		for (int j = 0; j < person.p_keypoints.size(); ++j){
			if (person.p_keypoints[j].p < 0.3){
				continue;
			}
			cv::circle(temp_img, cv::Point(person.p_keypoints[j].x, person.p_keypoints[j].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y), 2, cv::Scalar(0, 0, 255), -1);

		}
	}
	cv::imshow("Full Result", temp_img);
	char ch = cv::waitKey(0);
}

int main(int argc, char** argv) {
	std::string model_path = "../model/";
	//test_singleimage(model_path);
	//test_video(model_path);
	test_clear_img(model_path, "../test/2222.png");
	return 0;
}
