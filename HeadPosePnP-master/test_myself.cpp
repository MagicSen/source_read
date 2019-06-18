#include <iostream>
#include "RenderManager.h"
#include "glm.h"
#include "OGL_OCV_common.h"
#include <fstream>
#include "RenderEngine.h"

void loadWithPoints(Mat &op, Mat& ip, Mat &img, Mat &camMatrix, Mat &rvec, Mat &tvec) {
	int max_d = max(img.rows, img.cols);
	camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,
		0, max_d, img.rows / 2.0,
		0, 0, 1.0);
	//std::cout << "using cam matrix " << std::endl << camMatrix << std::endl;

	double _dc[] = { 0, 0, 0, 0 };
	solvePnP(op, ip, camMatrix, Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false, CV_EPNP);
}

bool loadNext(int &counter, cv::Mat &img, cv::Mat &ip) {
	printf("load %d\n", counter);

	const char* workingDir = "./";
	char buf[256] = { 0 };
	sprintf(buf, "%sAngelina_Jolie/Angelina_Jolie_%04d.txt", workingDir, counter);

	vector<Point2f > imagePoints;
	std::ifstream inputfile(buf);
	if (inputfile.fail()) {
		std::cerr << "can't read " << buf << std::endl; return false;
	}

	for (int i = 0; i<7; i++) {
		int x = 0, y = 0;
		inputfile >> std::skipws >> x >> y;
		imagePoints.push_back(Point2f((float)x, (float)y));
	}
	inputfile.close();

	ip = Mat(imagePoints).clone();
	sprintf(buf, "%sAngelina_Jolie/Angelina_Jolie_%04d.jpg", workingDir, counter);
	// 读取图像
	img = imread(buf);
	//loadWithPoints(ip, img);
	counter = (counter + 1);
	return true;
}

int testDraw3d2d(){
	// caculatePose
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

	Mat op = Mat(modelPoints);

	// 载入模型数据
	GLMmodel* head_obj = glmReadOBJ("head-obj.obj");
	GLMmodel* hat_obj = glmReadOBJ("new_hat.obj");
	cv::Mat icon_img = cv::imread("test.png");

	cv::Mat icon_img_2 = cv::imread("aa.png", CV_LOAD_IMAGE_UNCHANGED);
	std::cout << icon_img_2.channels() << std::endl;
	int width = 250, height = 250;

	RenderManager render_manager(width, height);
	if (!render_manager.isAvailable()){
		std::cout << "" << std::endl;
		return -2;
	}
	//cv::Mat img = cv::imread("Angelina_Jolie/Angelina_Jolie_0001.jpg");
	cv::Mat motion = (Mat_<double>(4, 4) << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	cv::Mat motion_hat = (Mat_<double>(4, 4) << 3.36717, 0, 0, 140.006,
		0, 3.36717, 0, -97.8008,
		0, 0, 3.36717, -68.3004,
		0, 0, 0, 1);

	glmTranslatePoint(hat_obj, motion_hat.ptr<double>());
	if (render_manager.enable()){
		cv::Mat img, ip;
		int counter = 1;
		cv::Mat k;
		vector<double> rv(3), tv(3);
		Mat rvec(rv), tvec(tv);
		while (loadNext(counter, img, ip)){
			loadWithPoints(op, ip, img, k, rvec, tvec);

			render_manager.render2DBackground(img);
			render_manager.render2DTexture(icon_img, cv::Point(25, 25));
			//render_manager.render2DTexture(icon_img_2, cv::Point(10, 10));

			//render_manager.render3DModel(head_obj, rvec, tvec, k);
			render_manager.render3DModel(hat_obj, rvec, tvec, k);
			std::cout << rvec << std::endl << tvec << std::endl << k << std::endl;
			cv::Mat render_img = render_manager.getRenderResult();
			cv::imshow("Render Image", render_img);
			cv::waitKey(0);
		}
	}
	render_manager.disable();
	return 0;
}
void testRender(){
	std::string material_file = "D:/project/source_read/HeadPosePnP-master/resource/resource.txt";
	std::string material_path = "D:/project/source_read/HeadPosePnP-master/resource/";
	int width = 250, height = 250;
	RenderEngine render_engine(material_file, material_path, width, height);
	if (render_engine.isAvailable()){
		cv::Mat img = cv::imread("D:/project/source_read/HeadPosePnP-master/Angelina_Jolie/Angelina_Jolie_0001.jpg");
		int id = 3;
		int status = -1;
		{
			// test 2d
			RenderParameter render_parameter;
			render_parameter.p_render_type = RenderType::DYNAMIC_2D_ANIMATION;
			render_parameter.p_material_id = 1;
			vector<int> start, end;
			start.push_back(25);
			start.push_back(25);
			end.push_back(100);
			end.push_back(100);
			render_parameter.p_render_start_position = start;
			render_parameter.p_render_end_position = end;
			render_parameter.p_animation_duration = 2.0;
			render_engine.setRenderParameter(render_parameter, id, status);

			while (1){
				Mat result = render_engine.renderFrame(img, status);
				cv::imshow("Render Image", result);
				char ch = cv::waitKey(30);
				if (ch == 'q'){
					break;
				}
				else if (ch == 'c'){
					render_engine.clearRenderParameter(id, status);
				}
				else if (ch == 'a'){
					render_engine.setRenderParameter(render_parameter, id, status);
				}
				else if (ch == 'u'){
					id += 1;
					render_engine.setRenderParameter(render_parameter, id, status);
				}
				else if (ch == 'p'){
					render_engine.clearAll();
				}
			}
		}

		{
			// test 3d
			cv::Mat rvec = (Mat_<double>(3, 1) << -2.907745739700069, 0.08962522108327355, -0.452754941357263);
			cv::Mat tvec = (Mat_<double>(3, 1) << -40.27142642338918, 91.0001081789147, 387.315340276434);
			cv::Mat k = (Mat_<double>(3, 3) << 250, 0, 125, 0, 250, 125, 0, 0, 1);
			RenderParameter render_parameter;
			render_parameter.p_render_type = RenderType::STATIC_3D_ANIMATION;
			render_parameter.p_material_id = 2;
			render_parameter.p_k = k;
			render_parameter.p_rvec = rvec;
			render_parameter.p_tvec = tvec;
			render_engine.setRenderParameter(render_parameter, id, status);
			Mat result = render_engine.renderFrame(img, status);
			cv::imshow("Render Image", result);
			char ch = cv::waitKey(0);
		}



	}
}

int main(int argc, char** argv)
{
	testRender();
	//testDraw3d2d();

	system("pause");
	return 0;
}