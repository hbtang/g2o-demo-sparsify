// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "stdafx.h"

#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <unordered_map>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
//#include "g2o/math_groups/se3quat.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "optimizer.h"
#include "converter.h"
#include "sample.h"
#include "simulator.h"

using namespace Eigen;
using namespace std;
using namespace odoslam;

int main(int argc, const char* argv[]){

    if (argc<2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] [STRUCTURE_ONLY] [DENSE]" << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)" << endl;
        cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)" << endl;
        cout << "STRUCTURE_ONLY: performe structure-only BA to get better point initializations (0 or 1; default: 0==false)" << endl;
        cout << "DENSE: Use dense solver (0 or 1; default: 0==false)" << endl;
        cout << endl;
        cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to 1==true." << endl;
        cout << endl;
        exit(0);
    }

    double PIXEL_NOISE = atof(argv[1]);
    double OUTLIER_RATIO = 0.0;

    if (argc>2)  {
        OUTLIER_RATIO = atof(argv[2]);
    }

    bool ROBUST_KERNEL = false;
    if (argc>3){
        ROBUST_KERNEL = atoi(argv[3]) != 0;
    }
    bool STRUCTURE_ONLY = false;
    if (argc>4){
        STRUCTURE_ONLY = atoi(argv[4]) != 0;
    }

    bool DENSE = false;
    if (argc>5){
        DENSE = atoi(argv[5]) != 0;
    }

    cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << endl;
    cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  endl;
    cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
    cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
    cout << "DENSE: " <<  DENSE << endl;


    // init simulator
    Simulator sim;
    sim.GenMeasSE3XYZ();
    sim.GenMeasXYZ2UV();

    // init optimizer
    SlamOptimizer optimizer;
    initOptimizer(optimizer, true);

    g2o::Isometry3D se3offset = g2o::Isometry3D::Identity();
    addParaSE3Offset(optimizer, se3offset, 0);

    // set vetex initial guess
    int vertexId = 0;

    // add KF vertex
    for(size_t i=0; i<sim.mvTrueKFs.size(); i++) {
        g2o::VertexSE3 * vSE3Tmp = new g2o::VertexSE3();
        vSE3Tmp->setId(vertexId);
        vSE3Tmp->setEstimate(sim.mvTrueKFs.at(i));
        if (i == 0) {
            vSE3Tmp->setFixed(true);
        }
        optimizer.addVertex(vSE3Tmp);
        vertexId ++;
    }

    // add MP vertex
    for(size_t i=0; i<sim.mvTrueMPs.size(); i++) {
        g2o::VertexPointXYZ* vXYZTmp = new g2o::VertexPointXYZ();

        vXYZTmp->setId(vertexId);
        vXYZTmp->setMarginalized(true);
        vXYZTmp->setEstimate(sim.mvTrueMPs.at(i)
                             + Vector3d(Sample::gaussian(0.1),
                                        Sample::gaussian(0.1),
                                        Sample::gaussian(0.1)));
        assert(optimizer.addVertex(vXYZTmp));

        vertexId ++;
    }

    // add edges
    for(size_t i=0; i<sim.mvMeasSE3XYZ.size(); i++) {

        int vertexIdMp = sim.mvMeasSE3XYZ.at(i).idMP + sim.mvTrueKFs.size();
        int vertexIdKF = sim.mvMeasSE3XYZ.at(i).idKF;

        g2o::Vector3D z = sim.mvMeasSE3XYZ.at(i).z;
        g2o::Matrix3d info = sim.mvMeasSE3XYZ.at(i).info;

        addEdgeSE3XYZ(optimizer, z, vertexIdKF, vertexIdMp, 0, info, 1);
    }

    // start g2o optimization
    cout << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(10);
    cout << endl;

    // results output

    cout << "cam 1: " << endl << toSE3Quat(estimateVertexSE3(optimizer, 1)) << endl;

//    for (int i = 15; i <= 30; i ++) {
//        g2o::Vector3D v_p = estimateVertexXYZ(optimizer, i);
//        Vector3d diff = v_p - sim.mvTrueMPs[i-15];
//        cout << diff.dot(diff) << endl;
//    }

}
