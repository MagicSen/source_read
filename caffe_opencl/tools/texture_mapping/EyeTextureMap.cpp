#include "EyeTextureMap.h"


EyeTextureMap::EyeTextureMap(std::string texture_file, std::string label_file) : BaseTextureMap(texture_file, label_file)
{
	// triangle 
	p_triangle.push_back(std::vector<int>{0, 1, 5});
	p_triangle.push_back(std::vector<int>{5, 1, 2});
	p_triangle.push_back(std::vector<int>{5, 2, 4});
	p_triangle.push_back(std::vector<int>{4, 2, 3});
	p_triangle.push_back(std::vector<int>{6, 7, 11});
	p_triangle.push_back(std::vector<int>{11, 7, 8});
	p_triangle.push_back(std::vector<int>{11, 8, 10});
	p_triangle.push_back(std::vector<int>{10, 8, 9});
	// face keypoints index
	p_mapping_src_index.push_back(36);
	p_mapping_src_index.push_back(37);
	p_mapping_src_index.push_back(38);
	p_mapping_src_index.push_back(39);
	p_mapping_src_index.push_back(40);
	p_mapping_src_index.push_back(41);
	p_mapping_src_index.push_back(42);
	p_mapping_src_index.push_back(43);
	p_mapping_src_index.push_back(44);
	p_mapping_src_index.push_back(45);
	p_mapping_src_index.push_back(46);
	p_mapping_src_index.push_back(47);

	// texture coordinate index
	p_mapping_texture_index.push_back(0);
	p_mapping_texture_index.push_back(1);
	p_mapping_texture_index.push_back(2);
	p_mapping_texture_index.push_back(3);
	p_mapping_texture_index.push_back(4);
	p_mapping_texture_index.push_back(5);
	p_mapping_texture_index.push_back(6);
	p_mapping_texture_index.push_back(7);
	p_mapping_texture_index.push_back(8);
	p_mapping_texture_index.push_back(9);
	p_mapping_texture_index.push_back(10);
	p_mapping_texture_index.push_back(11);
}


EyeTextureMap::~EyeTextureMap()
{
}