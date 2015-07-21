#include "OpticalFlow.h"
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/legacy/legacy.hpp>
#include <cv.h>

using namespace cv;
void test::video2imgs(std::string filename){
    int counter=0;
    VideoCapture capture(filename);
    Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;
        if(frame.empty())
            break;
        imshow("w", frame);
        std::stringstream temp;
        temp<<counter;
        imwrite("img/image"+temp.str()+".jpg",frame);
        counter++;
        waitKey(20); // waits to display frame
    }
    waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
}

Mat test::determineValues(const Mat& flow,test::motion& flag,float& thres){
    Mat values=Mat::zeros(flow.rows, flow.cols,CV_32S);
    switch(flag){
        case LEFT:
            for(int i=0;i<flow.rows;i++)
            {
                for(int j=0;j<flow.cols;j++)
                {
                    if(flow.at<Point2f>(i,j).x>thres)
                    {
                        values.at<int>(i,j)=1;
                    }
                }
            }
            break;
        case RIGHT:

            break;
        case UNKNOWN:
            break;
        case STAND:
            break;
    }
    return values;
}

std::vector<Point> test::detemineVehicle(const Mat& flow,const int& step,const test::motion& flag,const int& thres){
    std::vector<Point> res;
    switch(flag){
        case LEFT:
            for(int i=0;i<flow.rows;i++)
                for(int j=0;j<flow.cols;j++)
                {
                    bool isFeature = true;
                    for(int f=0;f<step;f++)
                        for(int g=0;g<step;g++)
                        {
                            if(i-f>=0&&j-g>=0)
                                if(flow.at<Point2f>(i-f,j-g).x>thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i-f>=0&&j+g<flow.cols)
                                if(flow.at<Point2f>(i-f,j+g).x>thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i+f<flow.rows&&j-g>=0)
                                if(flow.at<Point2f>(i+f,j-g).x>thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i+f<flow.rows&&j+g<flow.cols)
                                if(flow.at<Point2f>(i+f,j+g).x>thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                        }
                    if(isFeature) res.push_back(Point(j,i));
                }
        case RIGHT:
            for(int i=0;i<flow.rows;i++)
                for(int j=0;j<flow.cols;j++)
                {
                    bool isFeature = true;
                    for(int f=0;f<step;f++)
                        for(int g=0;g<step;g++)
                        {
                            if(i-f>=0&&j-g>=0)
                                if(flow.at<Point2f>(i-f,j-g).x<-thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i-f>=0&&j+g<flow.cols)
                                if(flow.at<Point2f>(i-f,j+g).x<-thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i+f<flow.rows&&j-g>=0)
                                if(flow.at<Point2f>(i+f,j-g).x<-thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                            if(i+f<flow.rows&&j+g<flow.cols)
                                if(flow.at<Point2f>(i+f,j+g).x<-thres)
                                {
                                    isFeature=false;
                                    break;
                                }
                        }
                    if(isFeature) res.push_back(Point(j,i));
                }
            break;
        case UNKNOWN:
            break;
        case STAND:
            break;
    }
    return res;
}


std::string test::motionEstimation(const Mat& flow, const Mat& img, float thres ){
    std::vector<std::vector<Point2f> > avg_flows;
    std::vector<Point2f> row;
    Size f_size = flow.size();
    int row_ws = f_size.height/4;
    int col_ws = f_size.width/4;
    //calculate data orientation value for the 8 regions
    for(int m=0;m<4;m++)
    {
        row.clear();
        for(int n=0;n<4;n++)
        {
            Point2f sum(0,0) ;
            int counter=0;
            sum.x=0;
            sum.y=0;
            for(int i=row_ws*m;i<row_ws*(m+1);i++)
            {
                for(int j=col_ws*n;j<col_ws*(n+1);j++)
                {
                    int gray = img.at<uchar>(i,j);
                    if(gray!=0&&gray!=255)
                    {
                        Point2f f_ij= flow.at<Point2f>(i,j);
                        sum += f_ij;
                        counter++;
                    }
                }
            }
            //std::cout<<"Position of vectors in "<<m<<" ,"<<n<<"; "<<std::endl;
            //std::cout<<sum.x/counter<<std::endl;
            //std::cout<<sum.y/counter<<std::endl;
            sum.x/=counter;
            sum.y/=counter;
            row.push_back(sum);
        }
        avg_flows.push_back(row);
    }
    //pack the result into a Mat
    for(int i=0;i<4;i++)
    {
        int num_left=0;
        int num_right=0;
        for(int j=0;j<4;j++)
        {
            Point2f point = avg_flows.at(i).at(j);
            if(point.x>thres)
            {
                num_left++;
            }
            else if(point.x<-thres)
            {
                num_right++;
            }
        }
        if(num_left==4) return "LEFT";
        if(num_right==4) return "RIGHT";
    }
    return "no result";
}

Mat test::computeFMatrix(const Mat& img,const Mat& flows, const int& ig_range, const int& upperBound, const int& lowerBound)
{
    std::vector<Point2f> cur(0);
    std::vector<Point2f> pre(0);
    for(int i=ig_range;i<flows.rows-ig_range;i++)
        for(int j=ig_range;j<flows.cols-ig_range;j++)
        {
            double pix_gray= img.at<double>(i,j);
            //sky and shadow area are noise
            if(pix_gray>lowerBound&&pix_gray<upperBound){
                Point2f flow=flows.at<Point2f>(j,i);
                pre.push_back(Point2f(double(i),double(j)));
                cur.push_back(Point2f(i+flow.x,j+flow.y));
            }

        }
    return findFundamentalMat(pre, cur, FM_RANSAC, 3, 0.99);

}


void test::tractVehicle(std::string path){
    VideoCapture cap(path);
    if( !cap.isOpened() )
        return;

    Mat prevgray, gray, flow, cflow, frame,resized;
    namedWindow("flow", 1);

    for(;;)
    {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if( prevgray.data )
        {
            //prev
            //next
            //flow
            //pyr_scale
            //levels: number of pyramid layers including the initial image
            //winsize: averaging window size
            //iterations: number of iterations
            //poly_n: size of the pixel neighborhood
            //poly_sigmaz: standard deviation(smooth factor)
            //flags:
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            string res=test::motionEstimation(flow,prevgray,1.0);
            std::vector<Point> features;
            if(res=="LEFT")
            {
                test::motion m=test::LEFT;
                features=test::detemineVehicle(flow,20,m,0);
            }
            else if(res=="RIGHT")
            {
                test::motion m=test::RIGHT;
                features=test::detemineVehicle(flow,20,m,0);
            }
            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
            test::drawOptFlowMap(flow, cflow, 1, 1.5, Scalar(0, 255, 0),res,features);
            resize(cflow,resized,cflow.size()*3);
            imshow("flow", resized);

        }
        if(waitKey(30)>=0)
            break;
        std::swap(prevgray, gray);
    }
}

void test::OpticalFlow(Mat pre,Mat cur,Mat& velx,Mat& vely){
    const  CvArr* prea=(CvArr*)&pre;
    const  CvArr* cura=(CvArr*)&cur;
    CvArr* x=(CvArr*)&velx;
    CvArr* y=(CvArr*)&vely;
    cvCalcOpticalFlowHS(prea,cura,0,x,y,0.5,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.3));

}

void test::drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color, const string& motion,std::vector<Point>& features){

    for(int y = 0; y < cflowmap.rows; y += step)
    {
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            //line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),color);
            //circle(cflowmap, Point(x,y), 0.5, color, -1);
        }
    }
    putText(cflowmap, motion, Point(cflowmap.rows/2,cflowmap.cols/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
    while(!features.empty())
    {
        Point p=features.back();
        if(p.x%16==0&&p.y%16==0)
        {
            circle(cflowmap,p,1,(0,0,255),-1);
        }
        features.pop_back();
    }
}

