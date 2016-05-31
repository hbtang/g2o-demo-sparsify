#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Config.h"

namespace odoslam{

typedef g2o::BlockSolverX SlamBlockSolver;
typedef g2o::LinearSolverCholmod<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg SlamAlgorithm;
typedef g2o::SparseOptimizer SlamOptimizer;
typedef g2o::CameraParameters CamPara;


void initOptimizer(SlamOptimizer &opt, bool verbose=false);
void addCamPara(SlamOptimizer &opt, const cv::Mat& K, int id);
void addVertexSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat& pose, int id, bool fixed=false);
void addVertexSBAXYZ(SlamOptimizer &opt, const Eigen::Vector3d &xyz, int id, bool marginal=true, bool fixed=false);

void addEdgeSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat& measure, int id0, int id1, const g2o::Matrix6d& info);
void addEdgeXYZ2UV(SlamOptimizer &opt, const Eigen::Vector2d& measure, int id0, int id1,
                   int paraId, const Eigen::Matrix2d &info, double thHuber);
g2o::SE3Quat estimateEdgeSE3ExpMapErr(const g2o::SE3Quat& estimate0,  const g2o::SE3Quat estimate1, const g2o::SE3Quat& measure);
g2o::Vector2D estimateEdgeXYZ2UVErr(const g2o::Vector2D &estimate0, const g2o::SE3Quat &estimate1, const cv::Mat& K, const cv::Point2f &measure);


void addParaSE3Offset(SlamOptimizer &opt, const g2o::Isometry3D& se3offset, int id);
void addVertexSE3(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id, bool fixed=false);
void addVertexXYZ(SlamOptimizer &opt, const g2o::Vector3D &xyz, int id, bool marginal=true);
void addEdgeSE3(SlamOptimizer &opt, const g2o::Isometry3D &measure, int id0, int id1, const g2o::Matrix6d& info);
void addEdgeSE3XYZ(SlamOptimizer &opt, const g2o::Vector3D& measure, int id0, int id1,
                   int paraSE3OffsetId, const g2o::Matrix3D &info, double thHuber);
g2o::Isometry3D estimateEdgeSE3Err(const g2o::Isometry3D& estimate0, const g2o::Isometry3D& estimate1, const g2o::Isometry3D &measure);
g2o::Vector3D estimateEdgeSE3XYZErr(const g2o::Isometry3D& v0, g2o::Vector3D &v1, const g2o::Isometry3D& parase3offset, const g2o::Vector3D& measure);

g2o::Isometry3D estimateVertexSE3(SlamOptimizer &opt, int id);
Eigen::Vector3d estimateVertexXYZ(SlamOptimizer &opt, int id);


g2o::SE3Quat estimateVertexSE3Expmap(SlamOptimizer &opt, int id);
g2o::Vector3D estimateVertexSBAXYZ(SlamOptimizer &opt, int id);

g2o::Matrix6d toSE3Info(const cv::Mat& mat);
g2o::Matrix6d toSE3Info(const g2o::Vector6d& vec6d);
g2o::Matrix3D toVec3dInfo(const cv::Point3f& xyzsigma);


}// namespace odoslam

#endif // OPTIMIZER_H
