#include "RenderEngine.h"
#include <fstream>


RenderAction::RenderAction(RenderParameter render_parameter, int id){
	p_render_parameter = render_parameter;
	p_id = id;
	if (p_render_parameter.p_render_type == DYNAMIC_2D_ANIMATION){
		p_frame_number = g_frame_rate * p_render_parameter.p_animation_duration;
	}
}


RenderEngine::RenderEngine(std::string material_resource_list, std::string material_resource_path, int width, int height) : p_width(width), p_height(height), p_render_manager(width, height)
{
	p_is_available = false;
	p_material_resource_list = material_resource_list;
	p_material_resource_path = material_resource_path;
	p_three_dimension_material.clear();
	p_two_dimension_material.clear();
	p_two_dimension_material_id_mapping.clear();
	p_three_dimension_material_id_mapping.clear();
	p_render_vector.clear();
	p_is_available = loadMaterialResource();
	p_is_available = p_render_manager.isAvailable();
}


RenderEngine::~RenderEngine()
{
}

bool RenderEngine::isAvailable(){
	return p_is_available;
}

bool RenderEngine::loadMaterialResource(){
	std::ifstream fin(p_material_resource_list);
	if (!fin.is_open()){
		return false;
	}
	std::string line;
	while (getline(fin,line)){
		if (line.length() < 3){
			continue;
		}
		std::stringstream ss(line);
		int material_type = -1;
		int material_id = -1;
		std::string material_path = "";
		ss >> material_type >> material_id >> material_path;

#if DEBUG_RENDER_ENGINE
		std::cout << line << std::endl;
		std::cout << material_type << " " << material_id << " " << material_path << std::endl;
#endif 
		if (material_type == -1 || material_id == -1){
			continue;
		}
		else if (material_type == 1){
			// load 2d texture
			cv::Mat texture = cv::imread(p_material_resource_path + material_path, CV_LOAD_IMAGE_UNCHANGED);
#if DEBUG_RENDER_ENGINE
			cv::imshow("Image", texture);
			cv::waitKey(0);
#endif
			p_two_dimension_material.push_back(texture.clone());
			int index = p_two_dimension_material.size() - 1;
			if (p_two_dimension_material_id_mapping.find(material_id) == p_two_dimension_material_id_mapping.end()){
				p_two_dimension_material_id_mapping[material_id] = index;
			}
			else{
				std::cerr << "Already has this id: " << material_type << " " << material_id << " " << material_path << std::endl;
			}

		}
		else if (material_type == 2){
			std::string full_material_path = (p_material_resource_path + material_path);
			char *three_model_path = new char[full_material_path.size() + 1];
			for (int i = 0; i < full_material_path.size(); ++i){
				three_model_path[i] = full_material_path[i];
			}
			three_model_path[full_material_path.size()] = '\0';
			GLMmodel* three_obj = glmReadOBJ(three_model_path);
			delete three_model_path;

			double matrix[16];
			for (int j = 0; j < 16; ++j){
				ss >> matrix[j];
			}
#if DEBUG_RENDER_ENGINE
			for (int k = 0; k < 16; ++k){
				std::cout << matrix[k] << " ";
			}
			std::cout << std::endl;
#endif
			glmTranslatePoint(three_obj, matrix);
			p_three_dimension_material.push_back(three_obj);
			int index = p_three_dimension_material.size() - 1;
			if (p_three_dimension_material_id_mapping.find(material_id) == p_three_dimension_material_id_mapping.end()){
				p_three_dimension_material_id_mapping[material_id] = index;
			}
			else{
				std::cerr << "Already has this id: " << material_type << " " << material_id << " " << material_path << std::endl;
			}
		}
	}
	fin.close();
	return true;
}


bool RenderEngine::setRenderParameter(RenderParameter render_parameter, int id, int &status){
	RenderAction render_action(render_parameter, id);
	p_render_vector.push_back(render_action);
	status = 0;
	return true;
}

bool RenderEngine::clearRenderParameter(int id, int &status){
	for (auto iter = p_render_vector.begin(); iter != p_render_vector.end();){
		if (iter->p_id == id){
			iter = p_render_vector.erase(iter);
		}
		else{
			++iter;
		}
	}
	status = 0;
	return true;
}

bool RenderEngine::clearAll(){
	p_render_vector.clear();
	return true;
}

cv::Mat RenderEngine::renderFrame(cv::Mat img, int &status){
	status = 0;
	cv::Mat result = img.clone();
	if (p_render_manager.isAvailable()){
		if (p_render_manager.enable()){
			p_render_manager.render2DBackground(img);

			for (int i = 0; i < p_render_vector.size(); ++i){
				RenderAction render_action = p_render_vector[i];
				if (render_action.p_render_parameter.p_render_type == RenderType::DYNAMIC_2D_ANIMATION){
					int total_frame = render_action.p_render_parameter.p_animation_duration * g_frame_rate;
					cv::Point2i position(0, 0);
					int current_frame = render_action.p_frame_number;
					if (current_frame >0){
						float lambda = (float(current_frame)) / total_frame;
						cv::Point2i start, end;
						start.x = render_action.p_render_parameter.p_render_start_position[0];
						start.y = render_action.p_render_parameter.p_render_start_position[1];
						end.x = render_action.p_render_parameter.p_render_end_position[0];
						end.y = render_action.p_render_parameter.p_render_end_position[1];
						position = (lambda)* start + (1 - lambda) * end;
						render_action.p_frame_number -= 1;
					}
					else{
						position.x = render_action.p_render_parameter.p_render_end_position[0];
						position.y = render_action.p_render_parameter.p_render_end_position[1];
					}

#if DEBUG_RENDER_ENGINE
					std::cout << position << " " << render_action.p_render_parameter.p_material_id << " " << p_two_dimension_material_id_mapping.size() << std::endl;
					for (auto iter = p_two_dimension_material_id_mapping.begin(); iter != p_two_dimension_material_id_mapping.end(); ++iter){
						std::cout << "map: " << iter->first << " " << iter->second << std::endl;
					}
#endif
					if (p_two_dimension_material_id_mapping.find(render_action.p_render_parameter.p_material_id) != p_two_dimension_material_id_mapping.end()){
						int index = p_two_dimension_material_id_mapping[render_action.p_render_parameter.p_material_id];
						cv::Mat texture = p_two_dimension_material[index];
						p_render_manager.render2DTexture(texture, position);	
					}
					else{
						status = 4;
					}
				}
				else if (render_action.p_render_parameter.p_render_type == RenderType::STATIC_3D_ANIMATION){
					if (p_three_dimension_material_id_mapping.find(render_action.p_render_parameter.p_material_id) != p_three_dimension_material_id_mapping.end()){
						int index = p_three_dimension_material_id_mapping[render_action.p_render_parameter.p_material_id];
						GLMmodel* model = p_three_dimension_material[index];
						p_render_manager.render3DModel(model, render_action.p_render_parameter.p_rvec, render_action.p_render_parameter.p_tvec, render_action.p_render_parameter.p_k);
					}
					else{
						status = 5;
					}
				}
				else{
					status = 3;
				}
				p_render_vector[i] = render_action;
			}
			result = p_render_manager.getRenderResult();
			p_render_manager.disable();
		}
		else{
			status = 2;
		}
	}
	else{
		status = 1;
	}
#if DEBUG_RENDER_ENGINE
	cv::imshow("temp1", result);
#endif
	return result;
}
