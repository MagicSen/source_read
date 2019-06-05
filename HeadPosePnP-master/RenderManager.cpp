#include "RenderManager.h"
#include "math.h"

RenderManager::RenderManager(int width, int height) : p_windows_render_source_manager(width, height)
{
	p_width = width;
	p_height = height;
	p_is_available = p_windows_render_source_manager.isAvailable();
	p_flag_texture = false;
}


RenderManager::~RenderManager()
{
}

bool RenderManager::init(){
	if (!p_flag_texture){
		p_img_with_drawing = MakeOpenCVGLTexture(Mat());
		p_flag_texture = true;
	}
	// 表示禁用多边形正面或者背面上的光照/阴影/颜色的计算
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// 设置着色模式，使用平滑的着色模式
	glShadeModel(GL_SMOOTH);

	// 启动深度缓冲区功能, 深度值发生变化进行比较后，符合条件的会绘制，此处小于等于参考值则通过
	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_LEQUAL);

	// 设置红绿蓝及alpha通道的混合方式，叠加方式：https://docs.microsoft.com/zh-cn/windows/desktop/OpenGL/glblendfunc
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LIGHT0);
	//// 启动光源
	glEnable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	return true;
}

void RenderManager::render2DBackground(cv::Mat img){
	p_img_with_drawing.set(img);
	glEnable2D();
	drawOpenCVImageInGL(p_img_with_drawing);
	glDisable2D();
}

void RenderManager::render2DTexture(cv::Mat img, cv::Point center){
	int w = img.cols;
	int h = img.rows;
	int left_w = center.x - w / 2;
	int left_h = center.y - h / 2;
	int right_w = center.x + w / 2;
	int right_h = center.y + h / 2;
	float x1 = left_w / p_width;
	float x2 = right_w / p_width;
	float y1 = left_h / p_height;
	float y2 = right_h / p_height;
	p_img_with_drawing.set(img);
	glEnable2D();
	drawOpenCVImageInGLWithIndex(p_img_with_drawing, x1, x2, y1, y2);
	glDisable2D();
}

void RenderManager::render3DModel(GLMmodel *model, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &k){
	cv::Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
	cv::Mat rotation;
	cv::Rodrigues(rvec, rotation);

	for (unsigned int row = 0; row<3; ++row)
	{
		for (unsigned int col = 0; col<3; ++col)
		{
			viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
		}
		viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
	}
	viewMatrix.at<double>(3, 3) = 1.0f;

	cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
	cvToGl.at<double>(0, 0) = 1.0f;
	cvToGl.at<double>(1, 1) = -1.0f; // Invert the y axis
	cvToGl.at<double>(2, 2) = -1.0f; // invert the z axis
	cvToGl.at<double>(3, 3) = 1.0f;
	viewMatrix = cvToGl * viewMatrix;
	cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64F);
	cv::transpose(viewMatrix, glViewMatrix);


	double fx = k.at<double>(0, 0);
	double fy = k.at<double>(1, 1);
	double height = k.at<double>(1, 2) * 2;
	double width = k.at<double>(0, 2) * 2;
	double fovy = 1.8 * atan(0.5*height / fy) * 180 / PI;
	double aspect = (width*fy) / (height*fx);

	// define the near and far clipping planes
	double view_near = 0.1;
	double view_far = 1000.0;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();

	glLoadIdentity();

	gluPerspective(fovy, aspect, view_near, view_far);
	glViewport(0, 0, width, height);


	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(glViewMatrix.ptr<double>());
	glmDraw(model, GLM_SMOOTH);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	SwapBuffers(p_windows_render_source_manager.getHDC());
}
//
//void RenderManager::render3DModel(GLMmodel *model, cv::Mat head_pose){
//
//	cv::Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
//	double tv[] = {-40.271, 91.000, 387.315};
//	double rot[] = {0.951, -0.029, 0.307, -0.090, -0.979, 0.184, 0.295, -0.203, -0.934};
//
//	for (unsigned int row = 0; row<3; ++row)
//	{
//		for (unsigned int col = 0; col<3; ++col)
//		{
//			viewMatrix.at<double>(row, col) = rot[row * 3 + col];
//		}
//		viewMatrix.at<double>(row, 3) = tv[row];
//	}
//	viewMatrix.at<double>(3, 3) = 1.0f;
//
//	cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
//	cvToGl.at<double>(0, 0) = 1.0f;
//	cvToGl.at<double>(1, 1) = -1.0f; // Invert the y axis
//	cvToGl.at<double>(2, 2) = -1.0f; // invert the z axis
//	cvToGl.at<double>(3, 3) = 1.0f;
//	viewMatrix = cvToGl * viewMatrix;
//	cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64F);
//	cv::transpose(viewMatrix, glViewMatrix);
//
//
//	double k[] = { 250, 0, 125, 0, 250, 125, 0, 0, 1};
//	double fx = k[0];
//	double fy = k[4];
//	double height = k[2] * 2;
//	double width = k[5] * 2;
//	double fovy = 1.8 * atan(0.5*height / fy) * 180 / PI;
//	double aspect = (width*fy) / (height*fx);
//
//	// define the near and far clipping planes
//	double view_near = 0.1;
//	double view_far = 1000.0;
//
//	glMatrixMode(GL_PROJECTION);
//	glPushMatrix();
//	glLoadIdentity();
//	
//	gluPerspective(fovy, aspect, view_near, view_far);
//	glViewport(0, 0, width, height);
//	//double project_old[16] = { 2.29984, 0, 0, 0, 0, 2.29994, 0, 0, 0, 0, -1.00002, -1, 0, 0, -0.0200002, 0 };
//	//glMultMatrixd(project_old);
//	glClear(GL_DEPTH_BUFFER_BIT);
//
//	glMatrixMode(GL_MODELVIEW);
//	glPushMatrix();
//	glLoadIdentity();
//	glLoadMatrixd(&glViewMatrix.at<double>(0, 0));
//
//
//	glmDraw(model, GLM_SMOOTH);
//
//	glPopMatrix();
//	glMatrixMode(GL_PROJECTION);
//	glPopMatrix();
//	SwapBuffers(p_windows_render_source_manager.getHDC());
//}

//void RenderManager::render3DModel(GLMmodel *model, cv::Mat head_pose){
//	glMatrixMode(GL_PROJECTION);
//	glPushMatrix();
//	glLoadIdentity();
//	double project_old[16] = { 2.29984, 0, 0, 0, 0, 2.29994, 0, 0, 0, 0, -1.00002, -1, 0, 0, -0.0200002, 0 };
//	glMultMatrixd(project_old);
//
//	double tv[] = {-40.271, 91.000, 387.315};
//	double rot[] = {0.951, -0.029, 0.307, -0.090, -0.979, 0.184, 0.295, -0.203, -0.934};
//
//
//	glClear(GL_DEPTH_BUFFER_BIT);
//	glMatrixMode(GL_MODELVIEW);
//	glPushMatrix();
//	glLoadIdentity();
//	// 设置观看的视角
//	gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
//	Vec3d tvv(tv[0], tv[1], tv[2]);
//	glTranslated(tvv[0], tvv[1], tvv[2]);
//
//	// rotate it
//	double _d[16] = { rot[0], rot[1], rot[2], 0,
//		rot[3], rot[4], rot[5], 0,
//		rot[6], rot[7], rot[8], 0,
//		0, 0, 0, 1 };
//	glMultMatrixd(_d);
//
//	glmDraw(model, GLM_SMOOTH);
//	glPopMatrix();
//	glMatrixMode(GL_PROJECTION);
//	glPopMatrix();
//	SwapBuffers(p_windows_render_source_manager.getHDC());
//}

cv::Mat RenderManager::getRenderResult(){
	GdiFlush();
	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
	Mat_<Vec3b> opengl_image(vPort[3], vPort[2]);
	{
		Mat_<Vec4b> opengl_image_4b(vPort[3], vPort[2]);
		glReadPixels(0, 0, vPort[2], vPort[3], GL_BGRA_EXT, GL_UNSIGNED_BYTE, opengl_image_4b.data);
		flip(opengl_image_4b, opengl_image_4b, 0);
		mixChannels(&opengl_image_4b, 1, &opengl_image, 1, &(Vec6i(0, 0, 1, 1, 2, 2)[0]), 3);
	}
	return opengl_image;
}

bool RenderManager::isAvailable(){
	return p_is_available;
}

bool RenderManager::enable(){
	if (p_is_available){
		p_is_available = p_windows_render_source_manager.enable();
		if (p_is_available){
			p_is_available = init();
			return p_is_available;
		}
		else{
			return false;
		}
	}
	else{
		return false;
	}
}

bool RenderManager::disable(){
	if (p_is_available){
		return p_windows_render_source_manager.disable();
	}
	return false;
}