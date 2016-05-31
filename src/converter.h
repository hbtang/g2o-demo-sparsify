#ifndef CONVERTER_H
#define CONVERTER_H

#include "Config.h"

namespace odoslam{

float header2Time(std_msgs::Header stamp);
float comparePose(const geometry_msgs::Pose& pose1, const geometry_msgs::Pose& pose2);
cv::Mat tf2R(tf::Transform tf);
cv::Mat tf2t(tf::Transform tf);
cv::Mat toT4x4(cv::Mat R, cv::Mat T);
cv::Mat toT4x4(float x, float y, float theta);
float compareRT(cv::Mat R1, cv::Mat R2, cv::Mat T1, cv::Mat T2);
tf::Transform slerpTf(tf::Transform tf, float v);

cv::Mat getR();
cv::Mat getT();
cv::Mat getRelativeR();
cv::Mat getRelativeT();
cv::Mat getRatTime(float time);
cv::Mat getTatTime(float time);
tf::Transform getTF();
tf::Transform getTFatTime(float time);
cv::Mat getRwithDuration(float duration);
cv::Mat getTwithDuration(float duration);
tf::Transform getTFwithDuration(float duration);
Eigen::Vector2d toVector2d(const cv::Point2f &cvVector);
g2o::Isometry3D toIsometry3D(const cv::Mat& T);
cv::Mat toCvMat(const g2o::Isometry3D& t);
cv::Point2f toCvPt2f(const Eigen::Vector2d& vec);
cv::Point3f toCvPt3f(const Eigen::Vector3d& vec);
g2o::Isometry3D toIsometry3D(const g2o::SE3Quat& se3quat);
g2o::SE3Quat toSE3Quat(const g2o::Isometry3D&  iso);
cv::Mat toMatSigmaCam(const XYTheta& dOdo);

// below from ORB_SLAM: https://github.com/raulmur/ORB_SLAM
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
cv::Mat toCvMat(const g2o::SE3Quat &SE3);
cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
cv::Mat toCvMat(const Eigen::Matrix3d &m);
cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
Eigen::Matrix<double,2,1> toVector2d(const cv::Mat &cvVector);
Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
std::vector<float> toQuaternion(const cv::Mat &M);

} // namespace odoslam
#endif
