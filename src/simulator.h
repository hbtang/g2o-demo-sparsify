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


class Simulator
{
public:
    Simulator();
    void Init();
    void GenMeasSE3XYZ();

    vector<Eigen::Vector3d> mvTrueMPs;
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > mvTrueKFs;
    vector<MeasSE3XYZ>  mvMeasSE3XYZ;
};



#endif // SIMULATOR_H
