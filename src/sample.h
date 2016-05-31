#ifndef SAMPLE_H
#define SAMPLE_H

#endif // SAMPLE_H

#include "stdafx.h"

//#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <unordered_map>

//#include "g2o/core/sparse_optimizer.h"
//#include "g2o/core/block_solver.h"
//#include "g2o/core/solver.h"
//#include "g2o/core/robust_kernel_impl.h"
//#include "g2o/core/optimization_algorithm_levenberg.h"
//#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
//#include "g2o/solvers/dense/linear_solver_dense.h"
//#include "g2o/types/sba/types_six_dof_expmap.h"
////#include "g2o/math_groups/se3quat.h"
//#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "optimizer.h"
#include "converter.h"

using namespace std;

class Sample {
public:
    static double uniform_rand(double lowerBndr, double upperBndr);
    static double gauss_rand(double mean, double sigma);
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};
