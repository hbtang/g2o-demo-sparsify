#ifndef SPARSIFIER_H
#define SPARSIFIER_H

#include "simulator.h"

class Sparsifier
{
public:
    Sparsifier();

    static void HessianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3d MP, MeasXYZ2UV measure, g2o::CameraParameters* pCamParam,
                              Eigen::Matrix<double, 9, 9>  & H );

    static void JacobianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3d MP, g2o::CameraParameters* pCamParam,
                               Eigen::Matrix<double, 2, 9>  & J);

    static void HessianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3d MP, const g2o::Matrix3d info,
                              Eigen::Matrix<double, 9, 9>  & H);

    static void JacobianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3d MP,
                               Eigen::Matrix<double, 3, 9>  & J);

    static void DoMarginalizeSE3XYZ(const vector<g2o::SE3Quat> vKF, const vector<g2o::Vector3d> vMP,
                                    const vector<MeasSE3XYZ> vMeasure,
                                    g2o::SE3Quat & z_out, g2o::Matrix6d & info_out);

    static void JacobianSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2,
                            Eigen::Matrix<double, 6, 12> & J);

    static void InfoSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2, const Eigen::Matrix<double, 12,12> & info,
                           Eigen::Matrix<double, 6, 6> & H);

};

#endif // SPARSIFIER_H
