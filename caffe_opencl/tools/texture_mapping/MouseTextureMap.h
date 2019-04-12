#pragma once
#include "BaseTextureMap.h"
class MouseTextureMap :
	public BaseTextureMap
{
public:
	MouseTextureMap(std::string texture_file, std::string label_file);
	~MouseTextureMap();
};

