#include "Config.h"

namespace odoslam{

std::string Config::DataPath;
int Config::ImgIndex;
cv::Size Config::ImgSize;
cv::Mat Config::bTc; // camera extrinsic
cv::Mat Config::cTb; // inv of bTc
cv::Mat Config::Kcam; // camera intrinsic
float Config::fxCam;
float Config::fyCam;
cv::Mat Config::Dcam; // camera distortion

float Config::UPPER_DEPTH;
float Config::LOWER_DEPTH;

int Config::NUM_FILTER_LAST_SEVERAL_MU;
int Config::FILTER_CONVERGE_CONTINUE_COUNT;
float Config::DEPTH_FILTER_THRESHOLD;

float Config::ScaleFactor; // scalefactor in detecting features
int Config::MaxLevel; // level number of pyramid in detecting features
int Config::MaxFtrNumber; // max feature number to detect
float Config::FEATURE_SIGMA;

float Config::ODO_X_UNCERTAIN, Config::ODO_Y_UNCERTAIN, Config::ODO_T_UNCERTAIN;
float Config::ODO_X_NOISE, Config::ODO_Y_NOISE, Config::ODO_T_NOISE;

int Config::LOCAL_FRAMES_NUM;
float Config::TH_HUBER;
int Config::LOCAL_ITER;

int Config::FPS;

cv::Mat Config::PrjMtrxEye;

void Config::readConfig(const std::string &path){
    DataPath = path;
    std::string camParaPath = path + "/config/Camyml";
    cv::FileStorage camPara(camParaPath, cv::FileStorage::READ);
    assert(camPara.isOpened());
    cv::Mat _mK, _mD, _rvec, rvec, _T, T, R;
    float height, width;
    camPara["image_height"] >> height;
    camPara["image_width"] >> width;
    camPara["camera_matrix"] >> _mK;
    camPara["distortion_coefficients"] >> _mD;
    camPara["rvec_b_c"] >> _rvec;
    camPara["tvec_b_c"] >> _T;
    _mK.convertTo(Kcam,CV_32FC1);
    _mD.convertTo(Dcam,CV_32FC2);
    _rvec.convertTo(rvec,CV_32FC1);
    _T.convertTo(T,CV_32FC1);
    fxCam = Kcam.at<float>(0,0);
    fyCam = Kcam.at<float>(1,1);
    ImgSize.height = height;
    ImgSize.width = width;
    std::cerr << "# Load camera config ..." << std::endl;
    std::cerr << "- Camera matrix: " << std::endl << " " <<
            Kcam << std::endl <<
            "- Camera distortion: " << std::endl << " " <<
            Dcam << std::endl <<
            "- Img size: " << std::endl << " " <<
            ImgSize << std::endl << std::endl;
    // bTc: camera extrinsic
    cv::Rodrigues(rvec,R);
    bTc = cv::Mat::eye(4,4,CV_32FC1);
    R.copyTo(bTc.rowRange(0,3).colRange(0,3));
    T.copyTo(bTc.rowRange(0,3).col(3));
    cv::Mat RT = R.t();
    cv::Mat t = -RT * T;
    cTb = cv::Mat::eye(4,4,CV_32FC1);
    RT.copyTo(cTb.rowRange(0,3).colRange(0,3));
    t.copyTo(cTb.rowRange(0,3).col(3));

    PrjMtrxEye = Kcam * cv::Mat::eye(4,4,CV_32FC1);
    camPara.release();

    std::string settingsPath = path + "/config/Settings.yml";
    cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
    assert(settings.isOpened());

    ImgIndex = (int)settings["img_num"];
    UPPER_DEPTH = (float)settings["upper_depth"];
    LOWER_DEPTH = (float)settings["lower_depth"];
    NUM_FILTER_LAST_SEVERAL_MU = (int)settings["depth_filter_avrg_count"];
    FILTER_CONVERGE_CONTINUE_COUNT = (int)settings["depth_filter_converge_count"];
    DEPTH_FILTER_THRESHOLD = (float)settings["depth_filter_thresh"];
    ScaleFactor = (float)settings["scale_facotr"];
    MaxLevel = (int)settings["max_level"];
    MaxFtrNumber = (int)settings["max_feature_num"];
    FEATURE_SIGMA = (float)settings["feature_sigma"];

    ODO_X_UNCERTAIN = (float)settings["odo_x_uncertain"];
    ODO_Y_UNCERTAIN = (float)settings["odo_y_uncertain"];
    ODO_T_UNCERTAIN = (float)settings["odo_theta_uncertain"];
    ODO_X_NOISE = (float)settings["odo_x_steady_noise"];
    ODO_Y_NOISE = (float)settings["odo_y_steady_noise"];
    ODO_T_NOISE = (float)settings["odo_theta_steady_noise"];
    LOCAL_FRAMES_NUM = (int)settings["frame_num"];
    TH_HUBER = sqrt((float)settings["th_huber2"]);
    LOCAL_ITER = (int)settings["local_iter"];
    FPS = (int)settings["fps"];

    settings.release();
}


XYTheta::XYTheta(){}
XYTheta::XYTheta(float _x, float _y ,float _theta):
    x(_x), y(_y), theta(_theta){}
XYTheta::~XYTheta(){}

XYTheta XYTheta::operator +(const XYTheta& toadd){
    // Note: dx and dy, which is expressed in the previous,
    // should be transformed to be expressed in the world frame
    float cost = std::cos(theta);
    float sint = std::sin(theta);
    float _x = x + toadd.x*cost - toadd.y*sint;
    float _y = y + toadd.x*sint + toadd.y*cost;
    float _theta = theta + toadd.theta;
    return XYTheta(_x, _y, _theta);
}

XYTheta XYTheta::operator -(const XYTheta& tominus){
    float dx = x - tominus.x;
    float dy = y - tominus.y;
    float dtheta = theta - tominus.theta;
    float cost = std::cos(tominus.theta);
    float sint = std::sin(tominus.theta);
    // Note: dx and dy, which is expressed in world frame,
    // should be transformed to be expressed in the previous frame
    return XYTheta(cost*dx+sint*dy, -sint*dx+cost*dy, dtheta);
}

}
