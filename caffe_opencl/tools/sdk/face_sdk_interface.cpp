#include "face_sdk_interface.h"

namespace face_sdk {
	FaceSDKBase::FaceSDKBase(std::string model_path)
	{
		p_face_sdk_implement = FaceSDKImplement(model_path);
		p_person_detail = std::vector<Person>();
	}

	FaceSDKBase::~FaceSDKBase()
	{
	}

	void FaceSDKBase::predict(float * img_data, int src_width, int src_height, bool is_color)
	{
		p_person_detail = p_face_sdk_implement.predict(img_data, src_width, src_height, is_color);
	}

	int FaceSDKBase::getNumberPeople()
	{
		return p_person_detail.size();
	}

	Nouse FaceSDKBase::getNouse(int index)
	{
		Nouse nouse;
		if (p_person_detail.size() == 0 || index > p_person_detail.size() - 1) {
			nouse.available = false;
			return nouse;
		}
		Person p = p_person_detail[index];

		PKeyPoint point = p.p_keypoints[30];
		point.x = point.x + p.p_rectangle.x;
		point.y = point.y + p.p_rectangle.y;
		nouse.nouse_pos = point;

		return nouse;
	}

	Eye FaceSDKBase::getEye(int index)
	{
		Eye eye;
		if (index > p_person_detail.size() - 1) {
			eye.available = false;
			return eye;
		}

		Person p = p_person_detail[index];

		return eye;
	}

	Ear FaceSDKBase::getEar(int index)
	{
		Ear ear;
		if (index > p_person_detail.size() - 1) {
			ear.available = false;
			return ear;
		}
		Person p = p_person_detail[index];
		return ear;
	}
	Mouse FaceSDKBase::getMouse(int index)
	{
		Mouse mouse;
		if (index > p_person_detail.size() - 1) {
			mouse.available = false;
			return mouse;
		}
		Person p = p_person_detail[index];
		return mouse;
	}

	Cheek FaceSDKBase::getCheek(int index)
	{
		Cheek cheek;
		if (index > p_person_detail.size() - 1) {
			cheek.available = false;
			return cheek;
		}
		Person p = p_person_detail[index];
		return cheek;
	}
	Person FaceSDKBase::getPersonKeypoints(int index){
		Person p;
		if (index > p_person_detail.size() - 1) {
			return p;
		}
		return p_person_detail[index];
	}
}
