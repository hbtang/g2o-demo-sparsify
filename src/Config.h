#ifndef CONFIG_H
#define CONFIG_H

#include "stdafx.h"


namespace odoslam{

const std::string PKG_PATH = "/home/fzheng/catkin_ws/src/odoslam/";

const float VERY_LARGE_FLOAT = 1000000.f;

const cv::Mat T4x4Eye = cv::Mat::eye(4,4,CV_32FC1);

struct XYTheta{
    float x;
    float y;
    float theta;
    XYTheta();
    XYTheta(float _x, float _y ,float _theta);
    ~XYTheta();
    XYTheta operator -(const XYTheta& tominus);
    XYTheta operator +(const XYTheta& toadd);
};

class WorkTimer
{
private:
    int64 tickBegin, tickEnd;
public:
    WorkTimer(){}
    ~WorkTimer(){}
    double time;
    void start(){
        tickBegin = cv::getTickCount();
    }

    void stop(){
        tickEnd = cv::getTickCount();
        time = (double)(tickEnd- tickBegin) / ((double)cv::getTickFrequency()) * 1000.;
    }
};


class Config{
public:
    static std::string DataPath;
    static int ImgIndex;
    static cv::Size ImgSize;
    static cv::Mat bTc; // camera extrinsic
    static cv::Mat cTb; // inv of bTc
    static cv::Mat Kcam; // camera intrinsic
    static float fxCam, fyCam;
    static cv::Mat Dcam; // camera distortion

    static float UPPER_DEPTH;
    static float LOWER_DEPTH;

    static int NUM_FILTER_LAST_SEVERAL_MU;
    static int FILTER_CONVERGE_CONTINUE_COUNT;
    static float DEPTH_FILTER_THRESHOLD;

    static float ScaleFactor; // scalefactor in detecting features
    static int MaxLevel; // level number of pyramid in detecting features
    static int MaxFtrNumber; // max feature number to detect
    static float FEATURE_SIGMA;

    static float ODO_X_UNCERTAIN, ODO_Y_UNCERTAIN, ODO_T_UNCERTAIN;
    static float ODO_X_NOISE, ODO_Y_NOISE, ODO_T_NOISE;

    static int LOCAL_FRAMES_NUM;
    static float TH_HUBER;
    static int LOCAL_ITER;

    static int FPS;
    static cv::Mat PrjMtrxEye;

    static void readConfig(const std::string& path);
    static bool acceptDepth(float depth);

};

}//namespace odoslam

#endif // CONFIG_H
