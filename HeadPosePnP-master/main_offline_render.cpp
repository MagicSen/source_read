#include <windows.h>
#include <iostream>
#include <gl/gl.h>
#include <gl/glu.h>
#include <string>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

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


int main(int argc, char* argv[])
{
	clock_t  clockBegin, clockEnd;
	const int WIDTH = 400;
	const int HEIGHT = 400;

	// Create a memory DC compatible with the screen
	HDC hdc = CreateCompatibleDC(0);
	if (hdc == 0) cout << "Could not create memory device context";

	// Create a bitmap compatible with the DC
	// must use CreateDIBSection(), and this means all pixel ops must be synchronised
	// using calls to GdiFlush() (see CreateDIBSection() docs)
	BITMAPINFO bmi = {
		{ sizeof(BITMAPINFOHEADER), WIDTH, HEIGHT, 1, 32, BI_RGB, 0, 0, 0, 0, 0 },
		{ 0 }
	};
	unsigned char *pbits; // pointer to bitmap bits
	HBITMAP hbm = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void **)&pbits,
		0, 0);
	if (hbm == 0) cout << "Could not create bitmap";

	//HDC hdcScreen = GetDC(0);
	//HBITMAP hbm = CreateCompatibleBitmap(hdcScreen,WIDTH,HEIGHT);

	// Select the bitmap into the DC
	HGDIOBJ r = SelectObject(hdc, hbm);
	if (r == 0) cout << "Could not select bitmap into DC";

	// Choose the pixel format
	PIXELFORMATDESCRIPTOR pfd = {
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
	int pfid = ChoosePixelFormat(hdc, &pfd);
	if (pfid == 0) cout << "Pixel format selection failed";

	// Set the pixel format
	// - must be done *after* the bitmap is selected into DC
	BOOL b = SetPixelFormat(hdc, pfid, &pfd);
	if (!b) cout << "Pixel format set failed";

	// Create the OpenGL resource context (RC) and make it current to the thread
	HGLRC hglrc = wglCreateContext(hdc);
	if (hglrc == 0) cout << "OpenGL resource context creation failed";
	wglMakeCurrent(hdc, hglrc);

	// Draw using GL - remember to sync with GdiFlush()
	clockBegin = clock();
	mGLRender();
	GdiFlush();
	//SaveBmp(hbm,"output.bmp");
	clockEnd = clock();
	printf("%d\n", clockEnd - clockBegin);

	clockBegin = clock();
	mGLRender1();
	GdiFlush();
	//SaveBmp(hbm,"output1.bmp");
	clockEnd = clock();
	printf("%d\n", clockEnd - clockBegin);
	/*
	Examining the bitmap bits (pbits) at this point with a debugger will reveal
	that the colored triangle has been drawn.
	*/

	//opencv show img
	Mat img(HEIGHT, WIDTH, CV_8UC4, (void *)pbits);
	imshow("img", img);
	waitKey();
	destroyWindow("img");

	// Clean up
	wglDeleteContext(hglrc); // Delete RC
	SelectObject(hdc, r); // Remove bitmap from DC
	DeleteObject(hbm); // Delete bitmap
	DeleteDC(hdc); // Delete DC

	system("pause");

	return 0;
}