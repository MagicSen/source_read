// HeadPose.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace cv;

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#elif defined(__linux__) || defined(__MINGW32__) || defined(WIN32)
#  include <GL/gl.h>
#  include <GL/glu.h>
#else
#  include <gl.h>
#endif

#include "glm.h"
#include "OGL_OCV_common.h"

void loadNext();
void loadWithPoints(Mat& ip, Mat& img);

const GLfloat light_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 0.0f, 0.0f, 1.0f, 0.0f };

const GLfloat mat_ambient[] = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

double rot[9] = { 0 };
GLuint textureID;
Mat backPxls;
vector<double> rv(3), tv(3);
Mat rvec(rv), tvec(tv);
Mat camMatrix;

//OpenCVGLTexture imgTex, imgWithDrawing;
OpenCVGLTexture imgWithDrawing;

GLMmodel* head_obj;
GLMmodel* hat_obj;


void myGLinit() {
	//    glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION ) ;

	// 表示禁用多边形正面或者背面上的光照/阴影/颜色的计算
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// 设置着色模式，使用平滑的着色模式
	glShadeModel(GL_SMOOTH);

	// 启动深度缓冲区功能, 深度值发生变化进行比较后，符合条件的会绘制，此处小于等于参考值则通过
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// 设置红绿蓝及alpha通道的混合方式，叠加方式：https://docs.microsoft.com/zh-cn/windows/desktop/OpenGL/glblendfunc
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//// 设置0号光源，启动法线，设置材质属性
	glEnable(GL_LIGHT0);
	//glEnable(GL_NORMALIZE);
	//glEnable(GL_COLOR_MATERIAL);
	//glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	//glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	//glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	//glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	//glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	//glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	//glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	//glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	//glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	//// 启动光源
	glEnable(GL_LIGHTING);
	// 更新MODELVIEW(世界坐标系下用来变化模型尺寸/平移/旋转)，同理可以使用GL_PROJECTION(定义投影方式：正交投影、视锥体)
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

void drawAxes() {
	// 绘制坐标系
	//Z = red
	glPushMatrix();
	glRotated(180, 0, 1, 0);
	glColor4d(1, 0, 0, 0.5);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0, 0, 1);
	glScaled(.1, .1, .1);
	//glutSolidTetrahedron();
	glPopMatrix();

	//Y = green
	glPushMatrix();
	glRotated(-90, 1, 0, 0);
	glColor4d(0, 1, 0, 0.5);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0, 0, 1);
	glScaled(.1, .1, .1);
	//glutSolidTetrahedron();
	glPopMatrix();

	//X = blue
	glPushMatrix();
	glRotated(-90, 0, 1, 0);
	glColor4d(0, 0, 1, 0.5);
	glBegin(GL_LINES);
	glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
	glEnd();
	glTranslated(0, 0, 1);
	glScaled(.1, .1, .1);
	//glutSolidTetrahedron();
	glPopMatrix();
}

Mat op;

void loadNext() {
	static int counter = 1;

	printf("load %d\n", counter);

	const char* workingDir = "./";

	char buf[256] = { 0 };
	sprintf(buf, "%sAngelina_Jolie/Angelina_Jolie_%04d.txt", workingDir, counter);

	vector<Point2f > imagePoints;
	ifstream inputfile(buf);
	if (inputfile.fail()) {
		cerr << "can't read " << buf << endl; return;
	}

	for (int i = 0; i<7; i++) {
		int x = 0, y = 0;
		inputfile >> skipws >> x >> y;
		imagePoints.push_back(Point2f((float)x, (float)y));
	}
	inputfile.close();

	Mat ip(imagePoints);

	sprintf(buf, "%sAngelina_Jolie/Angelina_Jolie_%04d.jpg", workingDir, counter);
	// 读取图像
	Mat img = imread(buf);
	// 设置背景纹理
	//imgTex.set(img); //TODO: what if different size??
	// 绘制关键点信息
	// paint 2D feature points
	for (unsigned int i = 0; i<imagePoints.size(); i++) circle(img, imagePoints[i], 2, Scalar(255, 0, 255), CV_FILLED);
	// 根据关键点信息计算头部姿态，将OpenCV下的变换坐标转为OpenGL下的变换坐标
	loadWithPoints(ip, img);
	// 设置输出纹理
	imgWithDrawing.set(img);
	// 计数器+1
	counter = (counter + 1);
}

void loadWithPoints(Mat& ip, Mat& img) {
	int max_d = MAX(img.rows, img.cols);
	camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,
		0, max_d, img.rows / 2.0,
		0, 0, 1.0);
	cout << "using cam matrix " << endl << camMatrix << endl;

	double _dc[] = { 0, 0, 0, 0 };
	solvePnP(op, ip, camMatrix, Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false, CV_EPNP);

	Mat rotM(3, 3, CV_64FC1, rot);
	Rodrigues(rvec, rotM);
	double* _r = rotM.ptr<double>();
	printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
		_r[0], _r[1], _r[2], _r[3], _r[4], _r[5], _r[6], _r[7], _r[8]);

	printf("trans vec: \n %.3f %.3f %.3f\n", tv[0], tv[1], tv[2]);

	double _pm[12] = { _r[0], _r[1], _r[2], tv[0],
		_r[3], _r[4], _r[5], tv[1],
		_r[6], _r[7], _r[8], tv[2] };

	Matx34d P(_pm);
	Mat KP = camMatrix * Mat(P);
	cout << "KP " << endl << KP << endl;

	//reproject object points - check validity of found projection matrix
	for (int i = 0; i<op.rows; i++) {
		Mat_<double> X = (Mat_<double>(4, 1) << op.at<float>(i, 0), op.at<float>(i, 1), op.at<float>(i, 2), 1.0);
		//		cout << "object point " << X << endl;
		Mat_<double> opt_p = KP * X;
		Point2f opt_p_img(opt_p(0) / opt_p(2), opt_p(1) / opt_p(2));
		//		cout << "object point reproj " << opt_p_img << endl; 
		// 绘制反投影点的坐标
		circle(img, opt_p_img, 4, Scalar(0, 0, 255), 1);
	}
	rotM = rotM.t();// transpose to conform with majorness of opengl matrix
}

class OfflineRender{
public:
	OfflineRender();
	~OfflineRender();
	void showErrorNumber();
	cv::Mat getRenderResult();
	bool isAvailable();
	HDC getHDC();
private:
	const int WIDTH = 250;
	const int HEIGHT = 250;
	// Create a memory DC compatible with the screen
	HDC p_hdc;

	HBITMAP p_hbm;
	unsigned char *p_pbits;
	// Select the bitmap into the DC
	HGDIOBJ p_r;

	// Choose the pixel format
	PIXELFORMATDESCRIPTOR p_pfd;

	// Create the OpenGL resource context (RC) and make it current to the thread
	HGLRC p_hglrc;
	int p_err_number;

};

OfflineRender::OfflineRender(){

	p_err_number = 0;
	p_pbits = NULL;
	// Create a memory DC compatible with the screen
	p_hdc = CreateCompatibleDC(0);
	if (p_hdc == 0) {
		p_err_number = 1;
	}
	else{
		// Create a bitmap compatible with the DC
		// must use CreateDIBSection(), and this means all pixel ops must be synchronised
		// using calls to GdiFlush() (see CreateDIBSection() docs)
		BITMAPINFO bmi = {
			{ sizeof(BITMAPINFOHEADER), WIDTH, HEIGHT, 1, 32, BI_RGB, 0, 0, 0, 0, 0 },
			{ 0 }
		};
		p_hbm = CreateDIBSection(p_hdc, &bmi, DIB_RGB_COLORS, (void **)&p_pbits,
			0, 0);
		if (p_hbm == 0) {
			p_err_number = 2;
		}
		else{
			// Select the bitmap into the DC
			p_r = SelectObject(p_hdc, p_hbm);
			if (p_r == 0) {
				p_err_number = 3;
			}
			else{
				// Choose the pixel format
				p_pfd = {
					sizeof(PIXELFORMATDESCRIPTOR), // struct size
					1, // Version number
					PFD_DRAW_TO_BITMAP | PFD_SUPPORT_OPENGL, // use OpenGL drawing to BM
					//PFD_DRAW_TO_BITMAP | PFD_SUPPORT_GDI | PFD_DOUBLEBUFFER,
					PFD_TYPE_RGBA, // RGBA pixel values
					32, // color bits
					0, 0, 0, // RGB bits shift sizes...
					0, 0, 0, // Don't care about them
					0, 0, // No alpha buffer info
					0, 0, 0, 0, 0, // No accumulation buffer
					32, // depth buffer bits
					0, // No stencil buffer
					0, // No auxiliary buffers
					PFD_MAIN_PLANE, // Layer type
					0, // Reserved (must be 0)
					0, // No layer mask
					0, // No visible mask
					0, // No damage mask
				};
				int pfid = ChoosePixelFormat(p_hdc, &p_pfd);
				if (pfid == 0) {
					p_err_number = 4;
				}
				else{
					// Set the pixel format
					// - must be done *after* the bitmap is selected into DC
					BOOL b = SetPixelFormat(p_hdc, pfid, &p_pfd);
					if (!b) {
						p_err_number = 5;
					}
					else{
						// Create the OpenGL resource context (RC) and make it current to the thread
						p_hglrc = wglCreateContext(p_hdc);
						if (p_hglrc == 0) {
							p_err_number = 6;
						}
						else{
							wglMakeCurrent(p_hdc, p_hglrc);
						}
					}
				}
			}
		}
	}
}


OfflineRender::~OfflineRender(){
	if (p_err_number == 0){
		// Clean up
		wglDeleteContext(p_hglrc); // Delete RC
		SelectObject(p_hdc, p_r); // Remove bitmap from DC
		DeleteObject(p_hbm); // Delete bitmap
		DeleteDC(p_hdc); // Delete DC
	}
	else if (p_err_number == 2){
		DeleteDC(p_hdc); // Delete DC
	}
	else if (p_err_number == 3){
		DeleteObject(p_hbm); // Delete bitmap
		DeleteDC(p_hdc); // Delete DC
	}
	else if (p_err_number == 4){
		SelectObject(p_hdc, p_r); // Remove bitmap from DC
		DeleteObject(p_hbm); // Delete bitmap
		DeleteDC(p_hdc); // Delete DC
	}
}

void OfflineRender::showErrorNumber(){
	switch (p_err_number)
	{
	case 0:
		std::cout << "Success" << std::endl;
		break;
	case 1:
		std::cout << "Could not create memory device context" << std::endl;
		break;
	case 2:
		std::cout << "Could not create bitmap" << std::endl;
		break;
	case 3:
		std::cout << "Could not select bitmap into DC" << std::endl;
		break;
	case 4:
		std::cout << "Pixel format selection failed" << std::endl;
		break;
	case 5:
		std::cout << "Pixel format set failed" << std::endl;
		break;
	case 6:
		std::cout << "OpenGL resource context creation failed" << std::endl;
		break;
	default:
		std::cout << "Unknown Error" << std::endl;
		break;
	}
}

cv::Mat OfflineRender::getRenderResult(){
	if (p_pbits != NULL){
		return cv::Mat(HEIGHT, WIDTH, CV_8UC4, (void *)p_pbits);
	}
	else{
		return cv::Mat();
	}
}

bool OfflineRender::isAvailable(){
	if (p_err_number == 0){
		return true;
	}
	else{
		return false;
	}
}

HDC OfflineRender::getHDC(){
	return p_hdc;
}

void mGLRender(HDC m_hDC)
{
	// 绘制图像
	glEnable2D();
	drawOpenCVImageInGL(imgWithDrawing);
	glDisable2D();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double project_old[16] = { 2.29984, 0, 0, 0, 0, 2.29994, 0, 0, 0, 0, -1.00002, -1, 0, 0, -0.0200002, 0 };
	glMultMatrixd(project_old);
	glClear(GL_DEPTH_BUFFER_BIT);
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		// 设置观看的视角
		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
		Vec3d tvv(tv[0], tv[1], tv[2]);
		glTranslated(tvv[0], tvv[1], tvv[2]);

		// rotate it
		double _d[16] = { rot[0], rot[1], rot[2], 0,
			rot[3], rot[4], rot[5], 0,
			rot[6], rot[7], rot[8], 0,
			0, 0, 0, 1 };
		glMultMatrixd(_d);

		glmDraw(head_obj, GLM_SMOOTH);
		glPopMatrix();
	}

	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		// 设置观看的视角
		glLoadIdentity();
		//double modelview_hat[16] = { 3.36717, 0, 0, 140.006, 0, 3.36717, 0, -97.8008, 0, 0, 3.36717, -68.3004, 0, 0, 0, 1 };
		//glMultMatrixd(modelview_hat);
		//double modelview_hat[16] = { 3.36717, 0, 0, 140.006, 0, 3.36717, 0, -97.8008, 0, 0, 3.36717, -68.3004, 0, 0, 0, 1 };
		//glMultMatrixd(modelview_hat);

		gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
		Vec3d tvv(tv[0], tv[1], tv[2]);
		glTranslated(tvv[0], tvv[1], tvv[2]);

		// rotate it
		double _d[16] = { rot[0], rot[1], rot[2], 0,
			rot[3], rot[4], rot[5], 0,
			rot[6], rot[7], rot[8], 0,
			0, 0, 0, 1 };
		glMultMatrixd(_d);

		glmDraw(hat_obj, GLM_SMOOTH);
		glPopMatrix();
	}


	//glScaled(50, 50, 50);

	//int draw_vPort[4]; glGetIntegerv(GL_VIEWPORT, draw_vPort);
	//for (int i = 0; i < 4; ++i){
	//	std::cout << draw_vPort[i] << " ";
	//}
	//std::cout << std::endl;

	//float modelview_matrix[16]; glGetFloatv(GL_MODELVIEW_MATRIX, modelview_matrix);
	//for (int i = 0; i < 16; ++i){
	//	std::cout << modelview_matrix[i] << " ";
	//}
	//std::cout << std::endl;
	//float projection_matrix[16]; glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix);
	//for (int i = 0; i < 16; ++i){
	//	std::cout << projection_matrix[i] << " ";
	//}
	//std::cout << std::endl;
	//drawAxes();
	//glPopMatrix();


	SwapBuffers(m_hDC);
}

int main(int argc, char** argv)
{

	// 得到模型关键点坐标
	vector<Point3f > modelPoints;
	//new model points: 
	modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
	modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
	modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
	modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
	modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695) 
	modelPoints.push_back(Point3f(-61.8886, 127.797, -89.4523));	// l ear (v 2011)
	modelPoints.push_back(Point3f(127.603, 126.9, -83.9129));		// r ear (v 1138)

	// 载入模型数据
	head_obj = glmReadOBJ("head-obj.obj");
	hat_obj = glmReadOBJ("new_hat.obj");
	op = Mat(modelPoints);
	Scalar m = mean(Mat(modelPoints));

	cout << "model points " << op << endl;

	// 初始化相机参数
	rvec = Mat(rv);
	double _d[9] = { 1, 0, 0,
		0, -1, 0,
		0, 0, -1 };
	Rodrigues(Mat(3, 3, CV_64FC1, _d), rvec);
	tv[0] = 0; tv[1] = 0; tv[2] = 1;
	tvec = Mat(tv);

	camMatrix = Mat(3, 3, CV_64FC1);

	OfflineRender offline_render;
	if (offline_render.isAvailable()){
		// 初始化opengl环境
		myGLinit();
		// 初始化OpenCV与OpenGL桥接的图像工具
		// prepare OpenCV-OpenGL images
		//imgTex = MakeOpenCVGLTexture(Mat());
		imgWithDrawing = MakeOpenCVGLTexture(Mat());
		bool flag = true;
		while (flag){
			loadNext();
			clock_t  clockBegin, clockEnd;
			clockBegin = clock();
			mGLRender(offline_render.getHDC());
			GdiFlush();
			clockEnd = clock();
			std::cout << "cost time: " << clockEnd - clockBegin << std::endl;
			//cv::Mat img = offline_render.getRenderResult();
			//cv::imshow("img", img);
			//cv::waitKey();
			{
				int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
				Mat_<Vec3b> opengl_image(vPort[3], vPort[2]);
				{
					Mat_<Vec4b> opengl_image_4b(vPort[3], vPort[2]);
					glReadPixels(0, 0, vPort[2], vPort[3], GL_BGRA_EXT, GL_UNSIGNED_BYTE, opengl_image_4b.data);
					flip(opengl_image_4b, opengl_image_4b, 0);
					mixChannels(&opengl_image_4b, 1, &opengl_image, 1, &(Vec6i(0, 0, 1, 1, 2, 2)[0]), 3);
				}
				cv::imshow("img render", opengl_image);
				char c = cv::waitKey();
				if (c == 'q'){
					flag = false;
				}
			}
		}
		//saveOpenGLBuffer();
	}
	else{
		offline_render.showErrorNumber();
	}
	return 0;
}


