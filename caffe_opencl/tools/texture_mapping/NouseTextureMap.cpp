#include "NouseTextureMap.h"


NouseTextureMap::NouseTextureMap(std::string texture_file, std::string label_file) : BaseTextureMap(texture_file, label_file)
{
	// triangle 
	p_triangle.push_back(std::vector<int>{4, 0, 1});
	p_triangle.push_back(std::vector<int>{4, 1, 2});
	p_triangle.push_back(std::vector<int>{4, 2, 3});
	p_triangle.push_back(std::vector<int>{4, 3, 5});
	p_triangle.push_back(std::vector<int>{5, 3, 6});
	p_triangle.push_back(std::vector<int>{6, 3, 7});
	p_triangle.push_back(std::vector<int>{7, 3, 8});
	p_triangle.push_back(std::vector<int>{8, 3, 2});
	p_triangle.push_back(std::vector<int>{8, 2, 1});
	p_triangle.push_back(std::vector<int>{8, 1, 0});

	// face keypoints index
	p_mapping_src_index.push_back(27);
	p_mapping_src_index.push_back(28);
	p_mapping_src_index.push_back(29);
	p_mapping_src_index.push_back(30);
	p_mapping_src_index.push_back(31);
	p_mapping_src_index.push_back(32);
	p_mapping_src_index.push_back(33);
	p_mapping_src_index.push_back(34);
	p_mapping_src_index.push_back(35);

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
}


NouseTextureMap::~NouseTextureMap()
{
}
