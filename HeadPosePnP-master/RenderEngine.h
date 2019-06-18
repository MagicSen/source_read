#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "glm.h"
#include <map>
#include "RenderManager.h"

#define DEBUG_RENDER_ENGINE 0
const int g_frame_rate = 15;
enum RenderType{
	DYNAMIC_2D_ANIMATION = 1,
	STATIC_2D_ANIMATION = 2,
	DYNAMIC_3D_ANIMATION = 3,
	STATIC_3D_ANIMATION = 4,
	DEFAULT_UNKNOWN = 5,
};

struct RenderParameter{
public:
	RenderParameter(){ p_material_id = -1; p_render_type = DEFAULT_UNKNOWN; p_render_start_position.clear(); p_render_end_position.clear();  p_animation_duration = 0; 
	p_k = (Mat_<double>(3, 3) << 250, 0, 125, 0, 250, 125, 0, 0, 1);
	p_rvec = (Mat_<double>(3, 1) << 0, 0, 0);
	p_tvec = (Mat_<double>(3, 1) << 0, 0, 0);
	}
	int p_material_id;
	RenderType p_render_type;
	std::vector<int> p_render_start_position;
	std::vector<int> p_render_end_position;
	float p_animation_duration;
	cv::Mat p_k;
	cv::Mat p_rvec;
	cv::Mat p_tvec;
};

struct RenderAction{
public:
	RenderAction(RenderParameter render_parameter, int id);
	RenderParameter p_render_parameter;
	int p_frame_number;
	int p_id;
};

class RenderEngine
{
public:
	RenderEngine(std::string material_resource_list, std::string material_resource_path, int width, int height);
	~RenderEngine();
	bool setRenderParameter(RenderParameter render_parameter, int id, int &status);
	bool clearRenderParameter(int id, int &status);
	bool clearAll();
	bool isAvailable();
	cv::Mat renderFrame(cv::Mat img, int &status);
private:
	bool loadMaterialResource();
	std::string p_material_resource_list;
	std::string p_material_resource_path;
	std::vector<GLMmodel*> p_three_dimension_material;
	std::vector<cv::Mat> p_two_dimension_material;
	std::map<int, int> p_two_dimension_material_id_mapping;
	std::map<int, int> p_three_dimension_material_id_mapping;
	bool p_is_available;
	std::vector<RenderAction> p_render_vector;
	RenderManager p_render_manager;
	const int p_width;
	const int p_height;
};

