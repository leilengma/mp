#ifndef MOTION_H_INCLUDED
#define MOTION_H_INCLUDED
#include <core/core.hpp>

cv::Mat SolveFundamental(double *pts1, double *pts2, const int& num_pts);

void determineMotion(cv::Mat f, cv::Mat& r, cv::Mat& t);


#endif // MOTION_H_INCLUDED
