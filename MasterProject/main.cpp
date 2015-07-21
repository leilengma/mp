#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "OpticalFlow.h"
#include "TwoPassArm.h"
#include "frame.h"
#include "vikit/pinhole_camera.h"
#include "feature_detection.h"
#include "initialization.h"
//linker option should include -lboost_system
#include "frame_handler_mono.h"
#include "initialization.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include "global.h"
#include "rrlib/object_tracking/tCMT.h"
using namespace cv;
using namespace std;


void static cameraPoseFromHomography(const Mat& H, Mat& pose)
{
    pose = Mat::eye(3, 4, CV_64F);      // 3x4 matrix, the camera pose
    double norm1 = (double)norm(H.col(0));
    double norm2 = (double)norm(H.col(1));
    double tnorm = (norm1 + norm2) / 2.0f; // Normalization value

    Mat p1 = H.col(0);       // Pointer to first column of H
    Mat p2 = pose.col(0);    // Pointer to first column of pose (empty)

    cv::normalize(p1, p2);   // Normalize the rotation, and copies the column to pose

    p1 = H.col(1);           // Pointer to second column of H
    p2 = pose.col(1);        // Pointer to second column of pose (empty)

    cv::normalize(p1, p2);   // Normalize the rotation and copies the column to pose

    p1 = pose.col(0);
    p2 = pose.col(1);

    Mat p3 = p1.cross(p2);   // Computes the cross-product of p1 and p2
    Mat c2 = pose.col(2);    // Pointer to third column of pose
    p3.copyTo(c2);       // Third column is the crossproduct of columns one and two

    pose.col(3) = H.col(2) / tnorm;  //vector t [R|t] is the last column of pose
}


void static calculatePos(Mat& prev, Mat& cur, Mat& t,Mat& r,rrlib::tracking::tCMT& tracker, Mat& camera){
    std::vector< cv::Point2f > pcorners,ncorners;
    std::vector< cv::Point2f > srcPoints,dstPoints;

    double qualityLevel = 0.01;
    double minDistance = 6;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    //detect all features
    cv::goodFeaturesToTrack(prev,pcorners, 400, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
    std::vector<unsigned char> status;
    std::vector<float> error;
    //calcOpticalFlowPyrLK(prev,cur, pcorners, ncorners, status, error);
    const double klt_win_size = 20.0;
    int klt_max_iter = 30;
    const double klt_eps = 0.001;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
    cv::calcOpticalFlowPyrLK(prev, cur, pcorners, ncorners, status, error);//,cv::Size2i(klt_win_size, klt_win_size),4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);


    //track the vehicle

    tracker.process_frame(cur);
    //filter out the car
    cv::line(cur, tracker.tl, tracker.br, Scalar(255,0,0));
    if(tracker.has_result){
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] == 0)
                continue;
            if(!(ncorners[i].y>tracker.tl.y&&ncorners[i].x>tracker.tl.x&&ncorners[i].y<tracker.br.y&&ncorners[i].x<tracker.br.x)){
                srcPoints.push_back(pcorners[i]);
                dstPoints.push_back(ncorners[i]);
            }
        }
    }else{
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] == 0)
                continue;
            srcPoints.push_back(pcorners[i]);
            dstPoints.push_back(ncorners[i]);
        }
    }
    for (int i=0;i<dstPoints.size();i++){
        circle(cur, dstPoints.at(i), 2, cv::Scalar(255,0,0), -1);
    }

    /*
    //calculate homography
    Mat h= findHomography(srcPoints,dstPoints, CV_RANSAC, 0.1);
    //std::cout<<h<<std::endl;
    Mat pos;
    cameraPoseFromHomography(h,pos);
    Mat rr = Mat::zeros(3,3,CV_64F);
    rr.at<double>(0,0)=pos.at<double>(0,0);
    rr.at<double>(0,1)=pos.at<double>(0,1);
    rr.at<double>(0,2)=pos.at<double>(0,2);
    rr.at<double>(1,0)=pos.at<double>(1,0);
    rr.at<double>(1,1)=pos.at<double>(1,1);
    rr.at<double>(1,2)=pos.at<double>(1,2);
    rr.at<double>(2,0)=pos.at<double>(2,0);
    rr.at<double>(2,1)=pos.at<double>(2,1);
    rr.at<double>(2,2)=pos.at<double>(2,2);
    std::cout<<rr<<std::endl;
    Mat ret;
    r=rr*r;
    cv::transpose(r,ret);
    Mat rt = pos.col(3);
    std::cout<<"=================================ret:"<<std::endl;
    std::cout<<ret*r<<std::endl;
    std::cout<<"=================================t:"<<std::endl;
    t=t+ret*rt;
    std::cout<<t<<std::endl;
    std::cout<<"=================================="<<std::endl;
    */


    //Mat eMatrix = cv::findEssentialMat(srcPoints,dstPoints,1,Point2d(0,0),RANSAC,0.99,1);
    Mat inliers;
    Mat fMatrix=findFundamentalMat(srcPoints, dstPoints, FM_LMEDS, 3,0.99,inliers);
    std::cout<<inliers<<std::endl;
    Mat w, u, vt;
    SVD::compute(fMatrix, w, u, vt);
    Mat z = cv::Mat::zeros(3, 3, CV_64F);
    z.at<double>(0,1) = 1.0;
    z.at<double>(1,0) =-1.0;
    Mat ww = cv::Mat::zeros(3, 3, CV_64F);
    ww.at<double>(0,1) = -1.0;
    ww.at<double>(1,0) = 1.0;
    ww.at<double>(2,2) = 1.0;
    Mat ut,ret;
    cv::transpose(u,ut);
    //extract translation
    Mat tx = u*z*ut;
    Mat rt = cv::Mat::zeros(3,1,CV_64F);
    //extract rotation
    Mat rr = u*ww*vt;
    r=rr*r;
    cv::transpose(rr,ret);
    std::cout<<"================relative rotation========="<<std::endl;
    std::cout<<rr*ret<<std::endl;
    cv::transpose(r,ret);
    std::cout<<"================absolute rotation=========="<<std::endl;
    std::cout<<r*ret<<std::endl;
    rt.at<double>(0,0)=tx.at<double>(2,1);
    rt.at<double>(1,0)=tx.at<double>(0,2);
    rt.at<double>(2,0)=tx.at<double>(1,0);
    t=t+ret*rt*5;


}


void static testOpticalflow(){
    Mat sample = imread("/home/fangsheng/mp/MasterProject/img/image2.jpg");
    sample.setTo(cv::Scalar(255,255,255));
    cv:Size s =sample.size()*3;
    Mat test;
    cv::resize(sample,test,s,sample.rows*3,sample.cols*3);
    Mat frame,gray,camera;
    Mat prevgray = imread("/home/fangsheng/mp/MasterProject/img/image54.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    gray = imread("/home/fangsheng/mp/MasterProject/img/image58.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    std::string path = "/home/fangsheng/mp/360_270_base.avi";
    Mat t = cv::Mat::zeros(3,1,CV_64F);
    Mat r = cv::Mat::eye(3,3,CV_64F);
    //initialize tracker
    rrlib::tracking::tCMT tracker=rrlib::tracking::tCMT();
    Point2f tl;
    tl.y=prevgray.rows/3-10;
    tl.x=prevgray.cols*6/11-15;
    Point2f br;
    br.y=prevgray.rows;
    br.x=prevgray.cols*18/19;
    tracker.init(prevgray,tl,br);
    cv::VideoCapture cap(path);
    //calculateTranslation(prevgray,gray,t,tracker,camera);
    //imshow("cur", gray);
    //waitKey(0);


    if(!cap.isOpened())
        return;
    for(;;){
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        calculatePos(prevgray,gray,t,r,tracker,camera);
        circle(test, Point(t.at<double>(0)+600,t.at<double>(1)+600), 1, cv::Scalar(255,0,0), -1);
        imshow("cur", gray);
        imshow("trj", test);
        waitKey(0);
        //if(waitKey(30)>=0)
        //    break;

    }


}

void static testsvo(){
    Mat prevgray, gray, flow, cflow, frame,resized, im1,im2,im3,im4,im5,im6;
    svo::initialization::KltHomographyInit init = svo::initialization::KltHomographyInit();
    gray     = imread("/home/fangsheng/mp/MasterProject/img/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE);

    vk::PinholeCamera cm = vk::PinholeCamera(640,480,1,1,0,0);
    svo::Vector3d prevP(70,70,70);
    /*
    prevgray = imread("/home/fangsheng/mp/MasterProject/img/image0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    gray     = imread("/home/fangsheng/mp/MasterProject/img/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im1     = imread("/home/fangsheng/mp/MasterProject/img/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im2     = imread("/home/fangsheng/mp/MasterProject/img/image2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im3     = imread("/home/fangsheng/mp/MasterProject/img/image3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im4     = imread("/home/fangsheng/mp/MasterProject/img/image4.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im5     = imread("/home/fangsheng/mp/MasterProject/img/image5.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im6     = imread("/home/fangsheng/mp/MasterProject/img/image6.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    */

    std::string path = "/home/fangsheng/mp/360_270_base.avi";
    Mat test = imread("/home/fangsheng/mp/MasterProject/img/image2.jpg");
    test.setTo(cv::Scalar(255,255,255));
    cv::VideoCapture cap(1);
    if( !cap.isOpened() )
        return;
    svo::FramePtr lf;
    svo::Vector3d p;
    svo::FrameHandlerMono handler(&cm);
    handler.start();
    namedWindow("flow", 1);
    double time = 0.0;
    std::vector<Point2f> history;
    for(;;)
    {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        handler.addImage(gray,time);
        lf= handler.lastFrame();
        p=lf->pos();
        std::ostringstream strs;
        strs<<"x:"<<p(0)<<",y:"<<p(1)<<"z:"<<p(2);
        std::cout<<"x:"<<p(0)<<",y:"<<p(1)<<"z:"<<p(2)<<std::endl;
        //std::cout<<"x:"<<p(0)<<",y:"<<p(1)<<"z:"<<p(2)<<std::endl;
        cv::circle(test,Point2f(p(0)+600,p(1)+600),2,Scalar(255,0,0),-1);
        std::string motion = strs.str();
        cv::putText(gray, motion, Point(gray.rows/2-15,gray.cols/2-15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
        time+=0.3;
        //resize(gray,resized,gray.size()*3);

        imshow("flow",gray);
        imshow("test", test);
        prevP=p;
        if(waitKey(30)>=0)
            break;
    }


    //svo::Frame ff(&cm,prevgray,0.0);
    //svo::Frame sf(&cm,gray,0.3);
    //svo::FramePtr fp(&ff);
    //svo::FramePtr sp(&sf);
    //init.addFirstFrame(fp);
    //init.addSecondFrame(sp);

    //handler.addImage(prevgray,0.0);
    //handler.setFirstFrame(fp);
    //handler.addImage(im1,0.3);
    //svo::FramePtr lf= handler.lastFrame();
    //svo::Vector3d p=lf->pos();
    //std::cout<<p<<std::endl;
    //handler.addImage(im2,0.6);
    //lf= handler.lastFrame();
    //p=lf->pos();
    //std::cout<<p<<std::endl;




}


void static smallTest(){
    Mat prevgray, gray, flow, cflow, frame,resized, im1,im2,im3,im4,im5,im6;
    prevgray = imread("/home/fangsheng/mp/MasterProject/img/image56.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    gray     = imread("/home/fangsheng/mp/MasterProject/img/image57.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im1     = imread("/home/fangsheng/mp/MasterProject/img/image58.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im2     = imread("/home/fangsheng/mp/MasterProject/img/image59.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im3     = imread("/home/fangsheng/mp/MasterProject/img/image60.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im4     = imread("/home/fangsheng/mp/MasterProject/img/image61.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im5     = imread("/home /fangsheng/mp/MasterProject/img/image62.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im6     = imread("/home/fangsheng/mp/MasterProject/img/image63.jpg",CV_LOAD_IMAGE_GRAYSCALE);

    svo::initialization::KltHomographyInit init = svo::initialization::KltHomographyInit();
    vk::PinholeCamera cm = vk::PinholeCamera(gray.cols,gray.rows,1,1,0,0);
    svo::FramePtr lf;
    svo::Vector3d p;
    svo::FrameHandlerMono handler(&cm);
    handler.start();
    handler.addImage(im1,0.0);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;
    handler.addImage(im2,0.3);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;
    handler.addImage(im3,0.6);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;
    handler.addImage(im4,0.9);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;
    handler.addImage(im5,1.2);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;
    handler.addImage(im6,1.5);
    lf= handler.lastFrame();
    p=lf->pos();
    std::cout<<p<<std::endl;






}

void static testMethode(){
    Mat prevgray, gray, flow, cflow, frame,resized, im1,im2,im3,im4,im5,im6;
    prevgray = imread("/home/fangsheng/mp/MasterProject/img/image0.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    gray     = imread("/home/fangsheng/mp/MasterProject/img/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im1     = imread("/home/fangsheng/mp/MasterProject/img/image1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im2     = imread("/home/fangsheng/mp/MasterProject/img/image2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im3     = imread("/home/fangsheng/mp/MasterProject/img/image3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im4     = imread("/home/fangsheng/mp/MasterProject/img/image4.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im5     = imread("/home/fangsheng/mp/MasterProject/img/image5.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    im6     = imread("/home/fangsheng/mp/MasterProject/img/image6.jpg",CV_LOAD_IMAGE_GRAYSCALE);

    std::vector< cv::Point2f > pcorners;
    std::vector< cv::Point2f > ncorners;

  // maxCorners – The maximum number of corners to return. If there are more corners
  // than that will be found, the strongest of them will be returned
    int maxCorners = 200;

  // qualityLevel – Characterizes the minimal accepted quality of image corners;
  // the value of the parameter is multiplied by the by the best corner quality
  // measure (which is the min eigenvalue, see cornerMinEigenVal() ,
  // or the Harris function response, see cornerHarris() ).
  // The corners, which quality measure is less than the product, will be rejected.
  // For example, if the best corner has the quality measure = 1500,
  // and the qualityLevel=0.01 , then all the corners which quality measure is
  // less than 15 will be rejected.
    double qualityLevel = 0.01;

  // minDistance – The minimum possible Euclidean distance between the returned corners
    double minDistance = 7.;

  // mask – The optional region of interest. If the image is not empty (then it
  // needs to have the type CV_8UC1 and the same size as image ), it will specify
  // the region in which the corners are detected
    cv::Mat mask;

    // blockSize – Size of the averaging block for computing derivative covariation
    // matrix over each pixel neighborhood, see cornerEigenValsAndVecs()
    int blockSize = 3;

    // useHarrisDetector – Indicates, whether to use operator or cornerMinEigenVal()
    bool useHarrisDetector = true;

    // k – Free parameter of Harris detector
    double k = 0.04;

    cv::goodFeaturesToTrack(prevgray, pcorners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    std::cout<<pcorners.size()<<std::endl;
    cv::goodFeaturesToTrack(gray, ncorners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    std::cout<<ncorners.size()<<std::endl;
    std::vector<Point2f> gcorners1, gcorners2;
    Mat fmatrix0=findFundamentalMat(pcorners, ncorners, FM_RANSAC, 3, 0.99);
    std::vector<Point2f> vfeatures(0);
    for(int i=0;i<pcorners.size();i++){
        Mat point=Mat::zeros(3,1,CV_64FC1);
        point.at<double>(0,0)=pcorners.at(i).x;
        point.at<double>(1,0)=pcorners.at(i).y;
        point.at<double>(2,0)=1.0;
        Mat epi = fmatrix0*point;
    }
    std::cout<<" "<<fmatrix0<<std::endl;
    //prev
    //next
    //flow
    //pyr_scale
    //levels: number of pyramid layers including the initial image
    //winsize: averaging window size
    //iterations: number of iterations
    //poly_n: size of the pixel neighborhood
    //poly_sigmaz: standard deviation(smooth factor)
    //flagps:
    //calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 10, 5, 3, 1.2, 0);
    //string res=test::motionEstimation(flow,prevgray,1.0);
    Mat fmatrix1=test::computeFMatrix(prevgray,flow,50,255,0);
    Mat fmatrix2=test::computeFMatrix(prevgray,flow,50,250,5);
    Mat fmatrix3=test::computeFMatrix(prevgray,flow,60,250,5);
    std::vector<svo::Vector3d> f_ref_;          //!< bearing vectors corresponding to the keypoints in the reference image.
    std::vector<svo::Vector3d> f_cur_;
    for( size_t i = 0; i < pcorners.size(); i++ )
    {
        if(pcorners[i].y<prevgray.rows/3.3){
            gcorners1.push_back(pcorners[i]);
            f_ref_.push_back(svo::Vector3d(pcorners[i].x,pcorners[i].y,1));
            cv::circle( prevgray, pcorners[i],3 , cv::Scalar( 255. ), -1 );
        }
        if(ncorners[i].y<gray.rows/3.3){
            f_cur_.push_back(svo::Vector3d(ncorners[i].x,ncorners[i].y,1));
            gcorners2.push_back(ncorners[i]);
        }
    }
    std::vector<int> inliers;
    std::vector<svo::Vector3d> xyz_in_cur;
    vk::PinholeCamera cm = vk::PinholeCamera(gray.cols,gray.rows,1,1,0,0);
    svo::Frame ff(&cm,prevgray,0.0);
    svo::Frame sf(&cm,gray,0.3);
    svo::FramePtr fp(&ff);
    svo::FramePtr sp(&sf);
    vector<double> disparities_;
    svo::initialization::trackKlt(fp,sp,gcorners1,gcorners2,f_ref_,f_cur_,disparities_);
    Sophus::SE3 T_cur_from_ref;
    //svo::initialization::computeHomography(f_ref_,f_cur_,1.0,10.0,inliers,T_cur_from_ref);
    if(gcorners1.size()>gcorners2.size())
    {
        while(gcorners1.size()!=gcorners2.size()) gcorners1.pop_back();
    }else{
        while(gcorners2.size()!=gcorners1.size()) gcorners2.pop_back();
    }
    std::cout<<gcorners1.size()<<std::endl;
    Mat fmatrixff=findFundamentalMat(gcorners1, gcorners2, FM_RANSAC, 3, 0.99);
    std::cout<<fmatrixff<<std::endl;
    cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
    //test::drawOptFlowMap(flow, cflow, 1, 1.5, Scalar(0, 255, 0),res,features);
    //std::cout<<" "<<fmatrix1<<std::endl;
    //std::cout<<" "<<fmatrix2<<std::endl;
    //std::cout<<" "<<fmatrix3<<std::endl;
    resize(cflow,resized,cflow.size()*3);
    imshow("flow", resized);
    waitKey(0);

}


int main(int, char**)
{
    //test::tractVehicle("/home/fangsheng/mp/360_270_base.avi");
    //testMethode();
    //testsvo();
    //testline();
    //smallTest();
    testOpticalflow();
    /*
    tp::lbSet *root10= new tp::lbSet;
    root10->label=1;
    tp::lbSet *root11= new tp::lbSet;
    root11->label=3;
    root10->next=root11;
    tp::lbSet *root12= new tp::lbSet;
    root12->label=-1;
    root11->next=root12;
    root12->next=root12;
    root11->tail=root11;
    root10->tail=root11;

    tp::lbSet *root20= new tp::lbSet;
    root20->label=2;
    tp::lbSet *root21= new tp::lbSet;
    root21->label=4;
    root20->next=root21;
    tp::lbSet *root22= new tp::lbSet;
    root22->label=-1;
    root21->next=root22;
    root22->next=root22;
    root21->tail=root21;
    root20->tail=root21;



    std::vector<tp::lbSet*> table;
    table.push_back(root10);
    table.push_back(root20);
    table.push_back(root10);
    table.push_back(root20);
    tp::resolve(1,3,table);
    //tp::mergeSet(table.at(3),table.at(0),table.at(3)->tail,table);
    while(root20->label!=-1)
    {
        int label=root20->label;
        std::cout<<label<<std::endl;
        root20=root20->next;
    }
    for(int i=0;i<table.size();i++)
    {
        int label=table.at(i)->label;
        std::cout<<label<<std::endl;
    }
    int *preRun=new int[10];
    cout<<*(preRun+5)<<endl;
    delete []preRun;
    */
    /*
    int data[64]={};
    Mat source = Mat::zeros(8,8,CV_32S);

    source.at<int>(0,0)=1;
    source.at<int>(0,1)=1;
    source.at<int>(0,3)=1;
    source.at<int>(0,4)=1;
    source.at<int>(0,7)=1;
    source.at<int>(1,1)=1;
    source.at<int>(1,2)=1;
    source.at<int>(1,3)=1;
    source.at<int>(1,6)=1;
    source.at<int>(1,7)=1;
    source.at<int>(2,4)=1;
    source.at<int>(2,5)=1;
    source.at<int>(3,7)=1;
    source.at<int>(3,0)=1;
    source.at<int>(3,1)=1;
    source.at<int>(3,2)=1;
    source.at<int>(4,2)=1;
    source.at<int>(4,3)=1;
    tp::calc2Pass(source);
    std::cout<<" "<<source<<std::endl;
    */
}



