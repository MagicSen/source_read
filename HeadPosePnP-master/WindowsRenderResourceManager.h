#pragma once
#include <Windows.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

class WindowsRenderResourceManager{
public:
	WindowsRenderResourceManager(int width, int height);
	~WindowsRenderResourceManager();
	void showErrorNumber();
	bool isAvailable();
	HDC getHDC();
	bool enable();
	bool disable();
private:
	int p_width;
	int p_height;
	// Create a memory DC compatible with the screen
	HDC p_hdc;
	HDC p_hdc_formal;
	HBITMAP p_hbm;
	unsigned char *p_pbits;
	// Select the bitmap into the DC
	HGDIOBJ p_r;

	// Choose the pixel format

	PIXELFORMATDESCRIPTOR p_pfd;

	// Create the OpenGL resource context (RC) and make it current to the thread
	HGLRC p_hglrc_formal;
	HGLRC p_hglrc;
	int p_err_number;
};