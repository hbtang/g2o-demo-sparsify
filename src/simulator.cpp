#include "simulator.h"
#include "sample.h"

using namespace std;
using namespace Eigen;

Simulator::Simulator() {
    Init();
    GenMeasSE3XYZ();
}

void Simulator::Init() {

    mvTrueMPs.clear();
    mvTrueKFs.clear();

    // initialize map points
    for (size_t i=0;i<500; ++i) {
        mvTrueMPs.push_back(Vector3d((Sample::uniform()-0.5)*3,
                                     Sample::uniform()-0.5,
                                     Sample::uniform()+3));
    }

    // initialize camera poses
    for (size_t i=0; i<15; ++i) {
        Vector3d trans(i*0.04-1.,0,0);

        Eigen::Quaterniond q;
        q.setIdentity();
        g2o::SE3Quat pose(q,trans);

        mvTrueKFs.push_back(pose);
    }
}

void Simulator::GenMeasSE3XYZ() {

    mvMeasSE3XYZ.clear();

    for (size_t i=0; i<mvTrueMPs.size(); i++) {
        for (size_t j=0; j<mvTrueKFs.size(); j++) {

            MeasSE3XYZ measTmp;
            measTmp.idKF = j;
            measTmp.idMP = i;
            measTmp.z = mvTrueKFs.at(j).inverse() * mvTrueMPs.at(i);
            measTmp.z += g2o::Vector3d(Sample::gaussian(0.05),
                                       Sample::gaussian(0.05),
                                       Sample::gaussian(0.05));
            measTmp.info = 0.05 * g2o::Matrix3d::Identity();

            mvMeasSE3XYZ.push_back(measTmp);
        }
    }
}
