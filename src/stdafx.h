#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <cassert>
#include <utility>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Float32MultiArray.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Image.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>


