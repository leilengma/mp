#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
namespace tp{
    struct lbSet{
        int label;
        lbSet *next;
        lbSet *tail;
    };

    void mergeSets(tp::lbSet *u, tp::lbSet *v, tp::lbSet *u_tail, std::vector<lbSet*>& table);
    void resolve(const int& x,const int& y,std::vector<lbSet*>& table);
    std::vector<int> calc2Pass(Mat& source);
    void detectRun(Mat& source, int row, std::vector<Point>& curRun);
    void labeling(Mat& source,const int& row, const std::vector<Point>& curRuns, const std::vector<Point>& preRuns,std::vector<lbSet*>& table, int& label);
}


