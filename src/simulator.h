#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "stdafx.h"

#include <Eigen/StdVector>


using namespace std;

struct MeasSE3XYZ {
    g2o::Vector3d z;
    g2o::Matrix3d info;
    int idMP = -1;
    int idKF = -1;
};

struct MeasXYZ2UV {
    g2o::Vector2d z;
    g2o::Matrix2d info;
    int idMP = -1;
    int idKF = -1;
};

struct MeasSE3Expmap {
    g2o::SE3Quat z;
    g2o::Matrix6d info;
    int id1 = -1;
    int id2 = -1;
};

class Simulator
{
public:
    Simulator();
    void Init();
    void GenMeasSE3XYZ();
    void GenMeasXYZ2UV();
    void GenMeasSE3Expmap();

    vector<Eigen::Vector3d> mvTrueMPs;
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > mvTrueKFs;

    vector<MeasSE3XYZ> mvMeasSE3XYZ;
    vector<MeasXYZ2UV> mvMeasXYZ2UV;
    vector<MeasSE3Expmap> mvMeasSE3Expmap;

    g2o::CameraParameters* mpCamParam;
};



#endif // SIMULATOR_H
