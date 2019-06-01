#include <windows.h>
#include <iostream>
#include <gl/gl.h>
#include <gl/glu.h>
#include <string>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class OfflineRender{
public:
	OfflineRender();
	~OfflineRender();
	void showErrorNumber();
	cv::Mat getRenderResult();
	bool isAvailable();
private:
	const int WIDTH = 400;
	const int HEIGHT = 400;
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
		std::cout <<  "Could not create bitmap" << std::endl;
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

void mGLRender()
{
	glClearColor(0.9f, 0.9f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0, 1.0, 1.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, -5, 0, 0, 0, 0, 1, 0);
	glBegin(GL_TRIANGLES);
	glColor3d(1, 0, 0);
	glVertex3d(0, 1, 0);
	glColor3d(0, 1, 0);
	glVertex3d(-1, -1, 0);
	glColor3d(0, 0, 1);
	glVertex3d(1, -1, 0);
	glEnd();
	glFlush(); // remember to flush GL output!
}

void mGLRender1()
{
	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0, 1.0, 1.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, -5, 0, 0, 0, 0, 1, 0);
	glBegin(GL_TRIANGLES);
	glColor3d(1, 0, 0);
	glVertex3d(0, 1, 0);
	glColor3d(0, 1, 0);
	glVertex3d(-1, -1, 0);
	glColor3d(0, 0, 1);
	glVertex3d(1, -1, 0);
	glEnd();
	glFlush(); // remember to flush GL output!
}

void test_offine(){
	OfflineRender offline_render;
	if (offline_render.isAvailable()){
		clock_t  clockBegin, clockEnd;
		clockBegin = clock();
		mGLRender();
		GdiFlush();
		clockEnd = clock();
		std::cout << "cost time: " << clockEnd - clockBegin << std::endl;
		cv::Mat img = offline_render.getRenderResult();
		cv::imshow("img", img);
		cv::waitKey();
	}
	else{
		offline_render.showErrorNumber();
	}
}

int main(int argc, char* argv[]){
	test_offine();
}