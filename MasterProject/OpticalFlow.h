#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <core/core.hpp>
using namespace cv;
namespace test
{
    enum behav{ACC=0,DEC=1,UNIF=2};
    enum motion{UNKNOWN=0,STAND=1,LEFT=2,RIGHT=3};
    void OpticalFlow(Mat pre,Mat cur,Mat& velx,Mat& vely);
    void tractVehicle(std::string path);
    void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color, const std::string& motion,std::vector<Point>& features);
    std::string motionEstimation(const Mat& flow,const Mat& img,float thres);
    std::vector<Point> detemineVehicle(const Mat& flow,const int& setp,const test::motion& flag,const int& thres);
    void video2imgs(std::string filename);
    std::vector<std::vector<Point> >  twoPassArm(Mat& flows);
    Mat determineValues(const Mat& flow,test::motion& flag,float& thres);
    Mat computeFMatrix(const Mat& img,const Mat& flows, const int& ig_range, const int& upperBound, const int& lowerBound);
}
