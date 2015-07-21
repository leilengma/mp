#include"Motion.h"

using namespace cv;

Mat SolveFundamental(double *pts1, double *pts2, const int& num_pts)
{
    assert(num_pts >= 8);

    if(num_pts < 8) {

    }

    // F is a temp variable, not the F fundamental matrix
    cv::Mat F(num_pts, 9, CV_64F);

    for(int i=0; i < num_pts; i++) {
        float x1 = pts1[i*2];
		float y1 = pts1[i*2+1];

		float x2 = pts2[i*2];
		float y2 = pts2[i*2+1];

        F.at<double>(i,0) = x1*x2;
        F.at<double>(i,1) = x2*y1;
        F.at<double>(i,2) = x2;
        F.at<double>(i,3) = x1*y2;
        F.at<double>(i,4) = y1*y2;
        F.at<double>(i,5) = y2;
        F.at<double>(i,6) = x1;
        F.at<double>(i,7) = y1;
        F.at<double>(i,8) = 1.0;
    }

    cv::SVD svd(F, cv::SVD::FULL_UV);

    double e00 = svd.vt.at<double>(5,0);
    double e01 = svd.vt.at<double>(5,1);
    double e02 = svd.vt.at<double>(5,2);
    double e03 = svd.vt.at<double>(5,3);
    double e04 = svd.vt.at<double>(5,4);
    double e05 = svd.vt.at<double>(5,5);
    double e06 = svd.vt.at<double>(5,6);
    double e07 = svd.vt.at<double>(5,7);
    double e08 = svd.vt.at<double>(5,8);

    double e10 = svd.vt.at<double>(6,0);
    double e11 = svd.vt.at<double>(6,1);
    double e12 = svd.vt.at<double>(6,2);
    double e13 = svd.vt.at<double>(6,3);
    double e14 = svd.vt.at<double>(6,4);
    double e15 = svd.vt.at<double>(6,5);
    double e16 = svd.vt.at<double>(6,6);
    double e17 = svd.vt.at<double>(6,7);
    double e18 = svd.vt.at<double>(6,8);

    double e20 = svd.vt.at<double>(7,0);
    double e21 = svd.vt.at<double>(7,1);
    double e22 = svd.vt.at<double>(7,2);
    double e23 = svd.vt.at<double>(7,3);
    double e24 = svd.vt.at<double>(7,4);
    double e25 = svd.vt.at<double>(7,5);
    double e26 = svd.vt.at<double>(7,6);
    double e27 = svd.vt.at<double>(7,7);
    double e28 = svd.vt.at<double>(7,8);

    double e30 = svd.vt.at<double>(8,0);
    double e31 = svd.vt.at<double>(8,1);
    double e32 = svd.vt.at<double>(8,2);
    double e33 = svd.vt.at<double>(8,3);
    double e34 = svd.vt.at<double>(8,4);
    double e35 = svd.vt.at<double>(8,5);
    double e36 = svd.vt.at<double>(8,6);
    double e37 = svd.vt.at<double>(8,7);
    double e38 = svd.vt.at<double>(8,8);
}

void determineMotion(Mat f, Mat& r, Mat& t)
{


}
