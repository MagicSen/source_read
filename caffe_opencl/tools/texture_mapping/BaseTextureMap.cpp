#include "BaseTextureMap.h"
#include <sstream>
#include <fstream>


bool isFileExist(std::string file_name){
	std::ifstream f(file_name);
	if (f){
		f.close();
		return true;
	}
	else{
		return false;
	}
}

std::vector<cv::Point2f> loadLabelFile(std::string label_file){
	std::vector<cv::Point2f> label_points;
	std::ifstream file(label_file);
	std::string line;
	while (getline(file, line))
	{
		std::istringstream s(line);
		float x, y;
		s >> x; s >> y;
		label_points.push_back(cv::Point2f(x, y));
	}
	file.close();
	return label_points;
}

BaseTextureMap::BaseTextureMap(std::string texture_file, std::string label_file)
{
	p_texture_type = Texture_Type_Common;
	if (!isFileExist(texture_file) || !isFileExist(label_file)){
		p_is_available = false;
	}
	else{
		// load texture image and points
		p_texture = cv::imread(texture_file);
		p_texture_points = loadLabelFile(label_file);
		p_is_available = true;
	}
}

bool BaseTextureMap::isAvailable(){
	return p_is_available;
}

BaseTextureMap::~BaseTextureMap()
{
}

cv::Mat BaseTextureMap::addMaterial(cv::Mat img, face_sdk::Person person){
	std::vector<cv::Point> total_src, total_texture;
	for (int i = 0; i < p_triangle.size(); ++i){
		bool flag = true;
		for (int j = 0; j < p_mapping_src_index.size(); ++j){
			if (person.p_keypoints[p_mapping_src_index[j]].p < 0.3){
				flag = false;
				break;
			}
		}
		if (!flag){
			continue;
		}
		if (p_triangle[i].size() != 3){
			continue;
		}
		std::vector<cv::Point2f> src_coord, texture_coord;
		src_coord.push_back(cv::Point2f(person.p_keypoints[p_mapping_src_index[p_triangle[i][0]]].x + person.p_rectangle.x, person.p_keypoints[p_mapping_src_index[p_triangle[i][0]]].y + person.p_rectangle.y));
		src_coord.push_back(cv::Point2f(person.p_keypoints[p_mapping_src_index[p_triangle[i][1]]].x + person.p_rectangle.x, person.p_keypoints[p_mapping_src_index[p_triangle[i][1]]].y + person.p_rectangle.y));
		src_coord.push_back(cv::Point2f(person.p_keypoints[p_mapping_src_index[p_triangle[i][2]]].x + person.p_rectangle.x, person.p_keypoints[p_mapping_src_index[p_triangle[i][2]]].y + person.p_rectangle.y));

		texture_coord.push_back(p_texture_points[p_mapping_texture_index[p_triangle[i][0]]]);
		texture_coord.push_back(p_texture_points[p_mapping_texture_index[p_triangle[i][1]]]);
		texture_coord.push_back(p_texture_points[p_mapping_texture_index[p_triangle[i][2]]]);

		std::vector<cv::Point> src_p, texture_p;
		getTriangleMapping(texture_coord, src_coord, texture_p, src_p);
		total_src.insert(total_src.end(), src_p.begin(), src_p.end());
		total_texture.insert(total_texture.end(), texture_p.begin(), texture_p.end());
	}
	// add texture
	cv::Mat add_information = img.clone();
	for (int i = 0; i < total_src.size(); ++i){
		cv::Point texture_cord = total_texture[i];
		cv::Point src_cord = total_src[i];
		add_information.at<cv::Vec3b>(src_cord) = p_texture.at<cv::Vec3b>(texture_cord);
	}
	return add_information;
}