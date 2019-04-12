#include "EyeBrowTextureMap.h"


EyeBrowTextureMap::EyeBrowTextureMap(std::string texture_file, std::string label_file) : BaseTextureMap(texture_file, label_file)
{
	// triangle 
	p_triangle.push_back(std::vector<int>{0, 1, 2});
	p_triangle.push_back(std::vector<int>{1, 2, 3});
	p_triangle.push_back(std::vector<int>{2, 3, 4});
	p_triangle.push_back(std::vector<int>{5, 6, 7});
	p_triangle.push_back(std::vector<int>{6, 7, 8});
	p_triangle.push_back(std::vector<int>{7, 8, 9});

	// face keypoints index
	p_mapping_src_index.push_back(17);
	p_mapping_src_index.push_back(18);
	p_mapping_src_index.push_back(19);
	p_mapping_src_index.push_back(20);
	p_mapping_src_index.push_back(21);
	p_mapping_src_index.push_back(22);
	p_mapping_src_index.push_back(23);
	p_mapping_src_index.push_back(24);
	p_mapping_src_index.push_back(25);
	p_mapping_src_index.push_back(26);

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
}


EyeBrowTextureMap::~EyeBrowTextureMap()
{
}
