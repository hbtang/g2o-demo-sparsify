#include "simulator.h"
#include "sample.h"

using namespace std;
using namespace Eigen;

Simulator::Simulator() {
    Init();

    double focalLength = 1000.;
    Vector2d principalPoint(320., 240.);
    mpCamParam = new g2o::CameraParameters(focalLength, principalPoint, 0.);
    mpCamParam->setId(0);
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
    for (size_t i=0; i<10; ++i) {
        Vector3d trans(i*0.05,0,0);

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

void Simulator::GenMeasXYZ2UV() {
    mvMeasXYZ2UV.clear();

    for (size_t i=0; i<mvTrueMPs.size(); i++) {
        for (size_t j=0; j<mvTrueKFs.size(); j++) {

            MeasXYZ2UV measTmp;
            measTmp.idKF = j;
            measTmp.idMP = i;

            measTmp.z = mpCamParam->cam_map(mvTrueKFs.at(j).map(mvTrueMPs.at(i)));
            measTmp.z += g2o::Vector2d(Sample::gaussian(1),
                                       Sample::gaussian(1));

            measTmp.info = 1.0 * g2o::Matrix2d::Identity();

            if (measTmp.z[0]>=0 && measTmp.z[1]>=0 &&
                    measTmp.z[0]<640 && measTmp.z[1]<480) {
                mvMeasXYZ2UV.push_back(measTmp);
            }
        }
    }
}

void Simulator::GenMeasSE3Expmap() {
    mvMeasSE3Expmap.clear();

    for (size_t i=0; i<mvTrueKFs.size()-1; i++) {

        MeasSE3Expmap measTmp;
        measTmp.id1 = i;
        measTmp.id2 = i+1;

        measTmp.z = mvTrueKFs.at(i).inverse() * mvTrueKFs.at(i+1);
        measTmp.info = 10000 * g2o::Matrix6d::Identity();

//        measTmp.info(3,3) = 1000;
//        measTmp.info(4,4) = 1000;
//        measTmp.info(5,5) = 1000;

        mvMeasSE3Expmap.push_back(measTmp);

//        cout << measTmp.info << endl;
    }
}















