#include "EarTextureMap.h"


EarTextureMap::EarTextureMap(std::string texture_file, std::string label_file) : BaseTextureMap(texture_file, label_file)
{
	// triangle 
	//p_triangle.push_back(std::vector<int>{0, 1, 2});
	//p_triangle.push_back(std::vector<int>{0, 1, 3});
	p_triangle.push_back(std::vector<int>{0, 3, 2});
	p_triangle.push_back(std::vector<int>{1, 2, 3});

	//p_triangle.push_back(std::vector<int>{4, 5, 6});
	//p_triangle.push_back(std::vector<int>{4, 5, 7});
	p_triangle.push_back(std::vector<int>{4, 7, 6});
	p_triangle.push_back(std::vector<int>{5, 6, 7});

	// face keypoints index
	p_mapping_src_index.push_back(68);
	p_mapping_src_index.push_back(69);
	p_mapping_src_index.push_back(70);
	p_mapping_src_index.push_back(71);
	p_mapping_src_index.push_back(72);
	p_mapping_src_index.push_back(73);
	p_mapping_src_index.push_back(74);
	p_mapping_src_index.push_back(75);


	// texture coordinate index
	p_mapping_texture_index.push_back(0);
	p_mapping_texture_index.push_back(1);
	p_mapping_texture_index.push_back(2);
	p_mapping_texture_index.push_back(3);
	p_mapping_texture_index.push_back(4);
	p_mapping_texture_index.push_back(5);
	p_mapping_texture_index.push_back(6);
	p_mapping_texture_index.push_back(7);
}


EarTextureMap::~EarTextureMap()
{
}

float minumProb(face_sdk::Person person, std::vector<int> p_index){
	float minum_prob = 1.0;
	for (int i = 0; i < p_index.size(); ++i){
		if (person.p_keypoints[p_index[i]].p < minum_prob){
			minum_prob = person.p_keypoints[p_index[i]].p;
		}
	}
	return minum_prob;
}

face_sdk::PKeyPoint changeToPKeyPoint(cv::Point p, float prob){
	face_sdk::PKeyPoint p_k;
	p_k.x = p.x;
	p_k.y = p.y;
	p_k.p = prob;
	return p_k;
}
face_sdk::Person EarTextureMap::updateEarPosition(face_sdk::Person person){

	//cv::Point p0 = cv::Point(person.p_keypoints[0].x, person.p_keypoints[0].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p16 = cv::Point(person.p_keypoints[16].x, person.p_keypoints[16].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p4 = cv::Point(person.p_keypoints[4].x, person.p_keypoints[4].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p12 = cv::Point(person.p_keypoints[12].x, person.p_keypoints[12].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p27 = cv::Point(person.p_keypoints[27].x, person.p_keypoints[27].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p30 = cv::Point(person.p_keypoints[30].x, person.p_keypoints[30].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);
	//cv::Point p33 = cv::Point(person.p_keypoints[33].x, person.p_keypoints[33].y) + cv::Point(person.p_rectangle.x, person.p_rectangle.y);

	cv::Point p0 = cv::Point(person.p_keypoints[0].x, person.p_keypoints[0].y);
	cv::Point p16 = cv::Point(person.p_keypoints[16].x, person.p_keypoints[16].y);
	cv::Point p4 = cv::Point(person.p_keypoints[4].x, person.p_keypoints[4].y);
	cv::Point p12 = cv::Point(person.p_keypoints[12].x, person.p_keypoints[12].y);
	cv::Point p27 = cv::Point(person.p_keypoints[27].x, person.p_keypoints[27].y);
	cv::Point p30 = cv::Point(person.p_keypoints[30].x, person.p_keypoints[30].y);
	cv::Point p33 = cv::Point(person.p_keypoints[33].x, person.p_keypoints[33].y);
	float new_prob = minumProb(person, std::vector<int>{0, 4, 12, 16, 27, 30, 33});
	float lambda = 0.75;
	cv::Point p0_near = lambda * p0 + (1 - lambda) * p16;
	cv::Point p16_near = (1 - lambda)* p0 + lambda * p16;

	float beta = 0.6;
	cv::Point p_left_center = p0_near + beta * (p0 - p4);
	cv::Point p_right_center = p16_near + beta * (p16 - p12);
	cv::Point p_temp = p27 - p30;
	float len = sqrt(p_temp.x * p_temp.x + p_temp.y * p_temp.y) / 2;
	cv::Point p_vertical = (p33 - p30);
	float len_v = sqrt(p_vertical.x * p_vertical.x + p_vertical.y * p_vertical.y);
	p_vertical.x = p_vertical.x / len_v * len * 2;
	p_vertical.y = p_vertical.y / len_v * len * 2;

	cv::Point p_horizon = p0 - p16;
	float len_h = sqrt(p_horizon.x * p_horizon.x + p_horizon.y * p_horizon.y);
	p_horizon.x = p_horizon.x / len_h * len;
	p_horizon.y = p_horizon.y / len_h * len;

	cv::Point p_up = p4 - p0;
	float len_u = sqrt(p_up.x * p_up.x + p_up.y * p_up.y);
	p_up.x = p_up.x / len_u * len * 0.5;
	p_up.y = p_up.y / len_u * len * 0.5;

	cv::Point p_left_1 = p_left_center + p_horizon;
	cv::Point p_left_2 = p_left_center - p_horizon;
	cv::Point p_left_3 = p_left_center - p_vertical;
	cv::Point p_left_4 = p_left_center - p_up;

	cv::Point p_right_1 = p_right_center + p_horizon;
	cv::Point p_right_2 = p_right_center - p_horizon;
	cv::Point p_right_3 = p_right_center - p_vertical;
	cv::Point p_right_4 = p_right_center - p_up;

	face_sdk::PKeyPoint pk_p;
	pk_p.x = p_left_1.x;
	person.p_keypoints.push_back(changeToPKeyPoint(p_left_1, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_left_2, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_left_3, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_left_4, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_right_1, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_right_2, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_right_3, new_prob));
	person.p_keypoints.push_back(changeToPKeyPoint(p_right_4, new_prob));
	return person;

}
cv::Mat EarTextureMap::addMaterial(cv::Mat img, face_sdk::Person person){
	person = updateEarPosition(person);
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
		if (src_cord.x >= img.cols || src_cord.y >= img.rows){
			continue;
		}
		add_information.at<cv::Vec3b>(src_cord) = p_texture.at<cv::Vec3b>(texture_cord);
	}
	return add_information;
}