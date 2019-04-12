#pragma once

#ifndef _TRIANGLE_MAPPING_TOOLS
#define _TRIANGLE_MAPPING_TOOLS

#include <iostream>
#include <vector>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

void getTriangleMapping(const std::vector<cv::Point2f> trangle_1, const std::vector<cv::Point2f> trangle_2, std::vector<cv::Point> &tr1, std::vector<cv::Point> &tr2);

#endif