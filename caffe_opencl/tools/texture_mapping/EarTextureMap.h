#pragma once
#include "BaseTextureMap.h"
class EarTextureMap :
	public BaseTextureMap
{
public:
	EarTextureMap(std::string texture_file, std::string label_file);
	~EarTextureMap();
	cv::Mat addMaterial(cv::Mat img, face_sdk::Person person);
	face_sdk::Person updateEarPosition(face_sdk::Person person);
};

