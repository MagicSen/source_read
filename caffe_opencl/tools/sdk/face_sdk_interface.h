#pragma once
#ifndef _FACE_SDK_INTERFACE
#define _FACE_SDK_INTERFACE
#include "face_sdk_implement.h"
#include <iostream>
#include <vector>

namespace face_sdk {
	struct Nouse {
		PKeyPoint nouse_pos;
		bool available;
	};

	struct Eye {
		PKeyPoint left_eye_pos;
		PKeyPoint right_eye_pos;
		bool available;
	};

	struct Ear {
		PKeyPoint left_ear_pos;
		PKeyPoint right_ear_pos;
		bool available;
	};
	
	struct Mouse {
		PKeyPoint mouse_pos;
		bool available;
	};

	struct Cheek {
		PKeyPoint left_cheek_pos;
		PKeyPoint right_cheek_pos;
		bool available;
	};

	class FaceSDKBase {
	public:
		FaceSDKBase(std::string model_path);
		~FaceSDKBase();
		void predict(float *img_data, int src_width, int src_height, bool is_color);
		int getNumberPeople();
	
		Nouse getNouse(int index);
		Eye getEye(int index);
		Ear getEar(int index);
		Mouse getMouse(int index);
		Cheek getCheek(int index);
		
		Person getPersonKeypoints(int index);

	private:
		FaceSDKImplement p_face_sdk_implement;
		std::vector<Person> p_person_detail;
	};
}

#endif