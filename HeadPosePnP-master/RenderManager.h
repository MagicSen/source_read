#pragma once
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "glm.h"
#include "WindowsRenderResourceManager.h"
#include "OGL_OCV_common.h"

#define PI 3.1415926
class RenderManager
{
public:
	RenderManager(int width, int height);
	~RenderManager();
	bool init();
	void render2DBackground(cv::Mat);
	void render2DTexture(cv::Mat, cv::Point);
	void render3DModel(GLMmodel *model, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &k);
	bool saveRenderResult();
	cv::Mat getRenderResult();
	bool isAvailable();
	bool enable();
	bool disable();
private:
	bool p_is_available;
	bool p_flag_texture;
	float p_width;
	float p_height;
	WindowsRenderResourceManager p_windows_render_source_manager;
	OpenCVGLTexture p_img_with_drawing;

};

