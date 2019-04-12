#include "triangle_mapping_tools.h"

float getTwoPointsDistance(cv::Point2f point1, cv::Point2f point2){
	return sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
}

float getStepLength(std::vector<cv::Point2f> trangle){
	float l1 = getTwoPointsDistance(trangle[0], trangle[1]);
	float l2 = getTwoPointsDistance(trangle[1], trangle[2]);
	float l3 = getTwoPointsDistance(trangle[2], trangle[0]);
	float l_max = (l1 > l2) ? (l1 > l3 ? l1 : l3) : (l2 > l3 ? l2 : l3);
	if (2 * l_max < 1){
		return 1.0;
	}
	else{
		return 1.0 / (2 * l_max);
	}
	
}

// trangle_1: known texture, trangle_2: not has texture
void getTriangleMapping(const std::vector<cv::Point2f> trangle_1, const std::vector<cv::Point2f> trangle_2, std::vector<cv::Point> &tr1, std::vector<cv::Point> &tr2){
	tr1.clear();
	tr2.clear();
	if (trangle_1.size() == trangle_2.size() && trangle_1.size() == 3){
		float step_length = getStepLength(trangle_2);
		cv::Point2f trangle_1_a = trangle_1[0];
		cv::Point2f trangle_1_b = trangle_1[1];
		cv::Point2f trangle_1_c = trangle_1[2];

		cv::Point2f trangle_2_a = trangle_2[0];
		cv::Point2f trangle_2_b = trangle_2[1];
		cv::Point2f trangle_2_c = trangle_2[2];

		for (float alpha = 0.0; alpha < 1.0; alpha += step_length){
			// trangle 1
			cv::Point2f p1 = (1 - alpha) * trangle_1_a + alpha * trangle_1_b;
			cv::Point2f p3 = (1 - alpha) * trangle_2_a + alpha * trangle_2_b;
			for (float beta = 0.0; beta < 1.0; beta += step_length){
				// trangle 1
				cv::Point2f p2 = (1 - beta) * p1 + beta * trangle_1_c;
				// trangle 2
				cv::Point2f p4 = (1 - beta) * p3 + beta * trangle_2_c;
				tr1.push_back(p2);
				tr2.push_back(p4);
			}
		}
	}
	else{
		return;
	}
}