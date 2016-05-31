#include "optimizer.h"
#include "converter.h"
#include "sugarCV.h"

namespace odoslam{

void initOptimizer(SlamOptimizer &opt, bool verbose){
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    opt.setAlgorithm(solver);
    opt.setVerbose(verbose);
}

void addCamPara(SlamOptimizer &opt, const cv::Mat &K, int id){
    Eigen::Vector2d principal_point(K.at<float>(0,2), K.at<float>(1,2));
    CamPara* campr = new CamPara(K.at<float>(0,0), principal_point, 0.);
    campr->setId(id);
    assert(opt.addParameter(campr));
}

void addParaSE3Offset(SlamOptimizer &opt, const g2o::Isometry3D& se3offset, int id){
    g2o::ParameterSE3Offset * para = new g2o::ParameterSE3Offset();
    para->setOffset(se3offset);
    para->setId(id);
    assert(opt.addParameter(para));
}

void addVertexSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat &pose, int id, bool fixed){
    g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
    v->setEstimate(pose);
    v->setFixed(fixed);
    v->setId(id);
    assert(opt.addVertex(v));
}

void addVertexSBAXYZ(SlamOptimizer &opt, const Eigen::Vector3d &xyz, int id, bool marginal, bool fixed){
    g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
    v->setEstimate(xyz);
    v->setId(id);
    v->setMarginalized(marginal);
    v->setFixed(fixed);
    assert(opt.addVertex(v));
}

void addVertexSE3(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id, bool fixed){
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setEstimate(pose);
    v->setFixed(fixed);
    v->setId(id);
    assert(opt.addVertex(v));
}

void addVertexXYZ(SlamOptimizer &opt, const g2o::Vector3D &xyz, int id, bool marginal){
    g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
    v->setEstimate(xyz);
    v->setId(id);
    v->setMarginalized(marginal);
    assert(opt.addVertex(v));
}

void addEdgeSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat &measure, int id0, int id1, const g2o::Matrix6d &info){
    g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
    e->setMeasurement(measure);
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setInformation(info);
    assert(opt.addEdge(e));
}

void addEdgeXYZ2UV(SlamOptimizer &opt, const Eigen::Vector2d &measure, int id0, int id1,
                   int paraId, const Eigen::Matrix2d &info, double thHuber){
    g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setMeasurement(measure);
    e->setInformation(info);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);
    e->setParameterId(0,paraId);
    assert(opt.addEdge(e));
}

g2o::SE3Quat estimateEdgeSE3ExpMapErr(const g2o::SE3Quat &estimate0, const g2o::SE3Quat estimate1, const g2o::SE3Quat &measure){
    return estimate1.inverse() * measure * estimate0;
}

g2o::Vector2D estimateEdgeXYZ2UVErr(const g2o::Vector3D &estimate0, const g2o::SE3Quat &estimate1, const cv::Mat& K, const cv::Point2f &measure){
    cv::Point3f pt3 = toCvPt3f(estimate1.map(estimate0));
    cv::Point2f pt2 = scv::prjcPt2Cam(K, cv::Mat::eye(4,4,CV_32FC1), pt3);
    g2o::Vector2D error = toVector2d(measure) - toVector2d(pt2);
    return error;
}

void addEdgeSE3(SlamOptimizer &opt, const g2o::Isometry3D &measure, int id0, int id1, const g2o::Matrix6d &info){
    g2o::EdgeSE3 *e =  new g2o::EdgeSE3();
    e->setMeasurement(measure);
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setInformation(info);
    assert(opt.addEdge(e));
}

void addEdgeSE3XYZ(SlamOptimizer &opt, const g2o::Vector3D &measure, int id0, int id1,
                   int paraSE3OffsetId, const g2o::Matrix3D &info, double thHuber){
    g2o::EdgeSE3PointXYZ* e = new g2o::EdgeSE3PointXYZ();
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setMeasurement(measure);
    e->setParameterId(0, paraSE3OffsetId);
    e->setInformation(info);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);
    assert(opt.addEdge(e));
}

g2o::Isometry3D estimateEdgeSE3Err(const g2o::Isometry3D& estimate0, const g2o::Isometry3D& estimate1, const g2o::Isometry3D &measure){
    return measure.inverse() * estimate0.inverse() * estimate1;
}


g2o::Vector3D estimateEdgeSE3XYZErr(const g2o::Isometry3D& v0, g2o::Vector3D &v1, const g2o::Isometry3D& parase3offset, const g2o::Vector3D& measure){
    g2o::Vector3D xyz = parase3offset.inverse() * v0.inverse() * v1;
    return xyz - measure;
}

g2o::Vector3D estimateVertexSBAXYZ(SlamOptimizer &opt, int id){
    g2o::VertexSBAPointXYZ* v = static_cast<g2o::VertexSBAPointXYZ*>
            (opt.vertex(id));
    return v->estimate();
}

g2o::SE3Quat estimateVertexSE3Expmap(SlamOptimizer &opt, int id){
    g2o::VertexSE3Expmap* v = static_cast<g2o::VertexSE3Expmap*>
            (opt.vertex(id));
    return v->estimate();
}

g2o::Isometry3D estimateVertexSE3(SlamOptimizer &opt, int id){
    g2o::VertexSE3 *v = static_cast<g2o::VertexSE3*>(opt.vertex(id));
    return v->estimate();
}

g2o::Vector3D estimateVertexXYZ(SlamOptimizer &opt, int id){
    g2o::VertexPointXYZ* v = static_cast<g2o::VertexPointXYZ*>(opt.vertex(id));
    return v->estimate();
}

g2o::Matrix6d toSE3Info(const cv::Mat &mat){
    g2o::Isometry3D iso = toIsometry3D(mat);
    return toSE3Info(g2o::internal::toVectorMQT(iso));
}

g2o::Matrix6d toSE3Info(const g2o::Vector6d &vec6d){
    g2o::Matrix6d ret = g2o::Matrix6d::Identity();
    // vec6d: x,y,z,qx,qy,qz
    for(unsigned i=0; i<3; i++){
        ret(i,i) = 1./(vec6d(i) * vec6d(i));
        ret(i,i) = ret(i,i)-1000<0? ret(i,i) : 1000;
        ret(i+3,i+3) = 1./(vec6d(i+3) * vec6d(i+3));
        ret(i+3,i+3) = ret(i+3,i+3)-100000<0? ret(i+3,i+3) : 100000;
    }
    return ret;
}

g2o::Matrix3D toVec3dInfo(const cv::Point3f &xyzsigma){
    g2o::Matrix3D ret = g2o::Matrix3D::Identity();
    ret(0,0) = 1.f/(xyzsigma.x * xyzsigma.x);
    ret(1,1) = 1.f/(xyzsigma.y * xyzsigma.y);
    ret(2,2) = 1.f/(xyzsigma.z * xyzsigma.z);
    for(unsigned i=0; i<3; i++){
        ret(i,i) = ret(i,i)>10000? 10000 : ret(i,i);
    }
    return ret;
}


}// namespace odoslam
