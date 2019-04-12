#include "MouseTextureMap.h"


MouseTextureMap::MouseTextureMap(std::string texture_file, std::string label_file) : BaseTextureMap(texture_file, label_file)
{
	// triangle 
	p_triangle.push_back(std::vector<int>{0, 1, 12});
	p_triangle.push_back(std::vector<int>{1, 2, 12});
	p_triangle.push_back(std::vector<int>{2, 12, 13});
	p_triangle.push_back(std::vector<int>{2, 13, 3});
	p_triangle.push_back(std::vector<int>{13, 3, 14});
	p_triangle.push_back(std::vector<int>{14, 3, 15});
	p_triangle.push_back(std::vector<int>{3, 4, 15});
	p_triangle.push_back(std::vector<int>{15, 4, 16});
	p_triangle.push_back(std::vector<int>{4, 5, 16});
	p_triangle.push_back(std::vector<int>{5, 16, 6});
	p_triangle.push_back(std::vector<int>{16, 6, 7});
	p_triangle.push_back(std::vector<int>{17, 16, 7});
	p_triangle.push_back(std::vector<int>{8, 17, 7});
	p_triangle.push_back(std::vector<int>{9, 17, 8});
	p_triangle.push_back(std::vector<int>{9, 18, 17});
	p_triangle.push_back(std::vector<int>{10, 18, 9});
	p_triangle.push_back(std::vector<int>{10, 19, 18});
	p_triangle.push_back(std::vector<int>{10, 12, 19});
	p_triangle.push_back(std::vector<int>{11, 12, 10});
	p_triangle.push_back(std::vector<int>{0, 12, 11});
	// face keypoints index
	p_mapping_src_index.push_back(48);
	p_mapping_src_index.push_back(49);
	p_mapping_src_index.push_back(50);
	p_mapping_src_index.push_back(51);
	p_mapping_src_index.push_back(52);
	p_mapping_src_index.push_back(53);
	p_mapping_src_index.push_back(54);
	p_mapping_src_index.push_back(55);
	p_mapping_src_index.push_back(56);
	p_mapping_src_index.push_back(57);
	p_mapping_src_index.push_back(58);
	p_mapping_src_index.push_back(59);
	p_mapping_src_index.push_back(60);
	p_mapping_src_index.push_back(61);
	p_mapping_src_index.push_back(62);
	p_mapping_src_index.push_back(63);
	p_mapping_src_index.push_back(64);
	p_mapping_src_index.push_back(65);
	p_mapping_src_index.push_back(66);
	p_mapping_src_index.push_back(67);


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
	p_mapping_texture_index.push_back(12);
	p_mapping_texture_index.push_back(13);
	p_mapping_texture_index.push_back(14);
	p_mapping_texture_index.push_back(15);
	p_mapping_texture_index.push_back(16);
	p_mapping_texture_index.push_back(17);
	p_mapping_texture_index.push_back(18);
	p_mapping_texture_index.push_back(19);
}


MouseTextureMap::~MouseTextureMap()
{
}
