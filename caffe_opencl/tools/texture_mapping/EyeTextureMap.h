#pragma once
#include "BaseTextureMap.h"
class EyeTextureMap :
	public BaseTextureMap
{
public:
	EyeTextureMap(std::string texture_file, std::string label_file);
	~EyeTextureMap();
};

