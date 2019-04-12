#pragma once
#include "BaseTextureMap.h"
class NouseTextureMap :
	public BaseTextureMap
{
public:
	NouseTextureMap(std::string texture_file, std::string label_file);
	~NouseTextureMap();
};

