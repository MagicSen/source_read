#pragma once
#ifndef _BASE_TEXTURE_MAP
#define _BASE_TEXTURE_MAP

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "face_sdk_implement.h"
#include "triangle_mapping_tools.h"

enum TextureType{
	Texture_Type_Common = 0,
	Texture_Type_Micromesh
};

class BaseTextureMap
{
public:
	BaseTextureMap(std::string texture_file, std::string label_file);
	~BaseTextureMap();
	virtual cv::Mat addMaterial(cv::Mat img, face_sdk::Person person);
	bool isAvailable();
public:
	cv::Mat p_texture;
	std::vector<std::vector<int>> p_triangle;
	std::vector<cv::Point2f> p_texture_points;
	std::vector<int> p_mapping_src_index;
	std::vector<int> p_mapping_texture_index;
	TextureType p_texture_type;
	bool p_is_available;
};

#endif
