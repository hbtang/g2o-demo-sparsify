#include "converter.h"

namespace odoslam{

tf::Transform slerpTf(tf::Transform tf, float v){
    tf::Quaternion q = tf::Quaternion::getIdentity().slerp(tf.getRotation(), v);
    tf::Vector3 t = tf::Vector3(0,0,0).lerp(tf.getOrigin(),v);
    return tf::Transform(q,t);
}

float header2Time(std_msgs::Header stamp){
    return float(stamp.stamp.sec)+float(stamp.stamp.nsec/100000)/10000.0;
}

cv::Mat tf2R(tf::Transform tf){
    cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);
    for(uint8_t i = 0; i < 3; i++){
        R.at<float>(i,0) = float(tf.getBasis().getRow(i).getX());
        R.at<float>(i,1) = float(tf.getBasis().getRow(i).getY());
        R.at<float>(i,2) = float(tf.getBasis().getRow(i).getZ());
    }
    return R.clone();
}

cv::Mat tf2t(tf::Transform tf){
    cv::Mat T = cv::Mat::zeros(3,1,CV_32FC1);
    T.at<float>(0,0) = float(tf.getOrigin().getX());
    T.at<float>(1,0) = float(tf.getOrigin().getY());
    T.at<float>(2,0) = float(tf.getOrigin().getZ());
    return T.clone();
}

cv::Mat toT4x4(cv::Mat R, cv::Mat T){
    cv::Mat T4x4 = cv::Mat::eye(4,4,R.type());
    R.copyTo(T4x4.rowRange(0,3).colRange(0,3));
    T.copyTo(T4x4.rowRange(0,3).col(3));
    return T4x4.clone();
}

cv::Mat toT4x4(float x, float y, float theta){
    float costht = cos(theta);
    float sintht = sin(theta);

    return (cv::Mat_<float>(4,4) <<
            costht,-sintht, 0, x,
            sintht, costht, 0, y,
            0,      0,      1, 0,
            0,      0,      0, 1);
}

float compareRT(cv::Mat R1, cv::Mat R2, cv::Mat T1, cv::Mat T2){
    // to be implement
    return 0;
}

float comparePose(const geometry_msgs::Pose &pose1, const geometry_msgs::Pose &pose2){
    float score = 0.0;
    float x1[7];
    float x2[7];
    x1[0] = pose1.position.x;
    x1[1] = pose1.position.y;
    x1[2] = pose1.position.z;
    x1[3] = pose1.orientation.w;
    x1[4] = pose1.orientation.x;
    x1[5] = pose1.orientation.y;
    x1[6] = pose1.orientation.z;
    x2[0] = pose2.position.x;
    x2[1] = pose2.position.y;
    x2[2] = pose2.position.z;
    x2[3] = pose2.orientation.w;
    x2[4] = pose2.orientation.x;
    x2[5] = pose2.orientation.y;
    x2[6] = pose2.orientation.z;
    for(uint8_t i = 0; i < 7; i++){
        score = score + fabs(x1[i]-x2[i]);
    }
    float total = 0.0;
    for(uint8_t i = 0; i < 7; i++){
        total = total + fabs(x1[i]);
    }
    return score/total;
}

Eigen::Vector2d toVector2d(const cv::Point2f &cvVector){
    Eigen::Vector2d v;
    v << cvVector.x, cvVector.y;
    return v;
}

g2o::Isometry3D toIsometry3D(const cv::Mat &T){
    Eigen::Matrix<double,3,3> R;
    R << T.at<float>(0,0), T.at<float>(0,1), T.at<float>(0,2),
         T.at<float>(1,0), T.at<float>(1,1), T.at<float>(1,2),
         T.at<float>(2,0), T.at<float>(2,1), T.at<float>(2,2);
    g2o::Isometry3D ret = (g2o::Isometry3D) Eigen::Quaterniond(R);
    Eigen::Vector3d t(T.at<float>(0,3), T.at<float>(1,3), T.at<float>(2,3));
    ret.translation() = t;
    return ret;
}

cv::Mat toCvMat(const g2o::Isometry3D &t){
    Eigen::Matrix<double,3,3> R = t.matrix().topLeftCorner<3,3>();
    Eigen::Matrix<double,3,1> T = t.translation();
    return toCvSE3(R,T);
}

cv::Point2f toCvPt2f(const Eigen::Vector2d& vec){
    return cv::Point2f(vec(0),vec(1));
}

cv::Point3f toCvPt3f(const Eigen::Vector3d& vec){
    return cv::Point3f(vec(0), vec(1), vec(2));
}

g2o::Isometry3D toIsometry3D(const g2o::SE3Quat &se3quat){
    return g2o::internal::fromSE3Quat(se3quat);
}

g2o::SE3Quat toSE3Quat(const g2o::Isometry3D &iso){
    return g2o::internal::toSE3Quat(iso);
}


// below from ORB_SLAM: https://github.com/raulmur/ORB_SLAM
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
            cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
            cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32FC1);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toMatSigmaCam(const XYTheta &dOdo){
    float dx = dOdo.x * Config::ODO_X_UNCERTAIN + Config::ODO_X_NOISE;
    float dy = dOdo.y * Config::ODO_Y_UNCERTAIN + Config::ODO_Y_NOISE;
    float dtheta = dOdo.theta * Config::ODO_T_UNCERTAIN + Config::ODO_T_NOISE;

    cv::Mat dbTb = toT4x4(dx, dy, dtheta);
    return Config::cTb * dbTb * Config::bTc;
}

cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32FC1);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32FC1);
    for(int i=0;i<3;i++)
        cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32FC1);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,2,1> toVector2d(const cv::Mat &cvVector)
{
    ROS_ASSERT((cvVector.rows==2 && cvVector.cols==1) || (cvVector.cols==2 && cvVector.rows==1));

    Eigen::Matrix<double,2,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1);

    return v;
}

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

}
