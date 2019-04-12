#pragma once

#ifndef _FACE_SDK_IMPLEMENT
#define _FACE_SDK_IMPLEMENT

#include <iostream>
#include <vector>
#include <string>
#include "head.h"

namespace face_sdk {

	struct PKeyPoint {
		float x;
		float y;
		float p;
		PKeyPoint() {
			x = 0;
			y = 0;
			p = 0;
		}
	};

	struct PRectangle {
		float x;
		float y;
		float w;
		float h;
		PRectangle() {
			x = 0;
			y = 0;
			w = 0;
			h = 0;
		}
	};

	struct Person {
		PRectangle p_rectangle;
		std::vector<PKeyPoint> p_keypoints;
		bool isAvailable(){ if (p_keypoints.size() != 68) return false; else return true;}
	};

	class FaceSDKImplement {
	public:
		// model initinal
		FaceSDKImplement();
		FaceSDKImplement(std::string model_path);
		~FaceSDKImplement();
		bool init();
		bool isAvailable();

		std::vector<Person> predict(float *img_data, int src_width, int src_height, bool is_color);
	private:
		bool p_isAvailable;
		std::string p_modelPath;
		void *caffe_net_detection;
		void *caffe_net_keypoints;
	};
}

#endif
