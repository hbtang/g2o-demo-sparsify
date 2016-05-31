// sugarCV: tool functions for opencv
// By ZHEGN Fan fzheng@link.cuhk.edu.hk

// Use namespace: scv (Sugar CV)

#ifndef SUGAR_CV_H
#define SUGAR_CV_H

#include "stdafx.h"

namespace scv{

const float BW = 5.f;
const float L = 15.f;
const float INRATIO = 0.996f;
const float dr = 1.f;

using std::vector;
using namespace cv;

Mat inv(const Mat& T4x4);

void pts2Ftrs(const vector<KeyPoint>& _orgnFtrs,
              const vector<Point2f>& _points, vector<KeyPoint>& _features);


Mat drawKeys(const vector<KeyPoint> keys, const Mat &img,
             vector<uchar> mask = vector<uchar>(0, 0));

Mat drawKeysWithNum(const vector<KeyPoint> keys, const Mat &img,
             vector<uchar> mask = vector<uchar>(0, 0));

Mat drawQueryKeys(const vector<KeyPoint> keys, const Mat &img,
                  const vector<DMatch>& matches,
                  vector<uchar> mask = vector<uchar>(0, 0));

Mat drawMatchesInOneImg(const vector<KeyPoint> queryKeys,
                        const Mat &trainImg, const vector<KeyPoint> trainKeys,
                        const vector<DMatch> &matches, vector<uchar> mask = vector<uchar>(0, 0));

Mat drawMatchesInOneImg(const vector<KeyPoint> queryKeys,
                        const Mat &trainImg, const vector<KeyPoint> trainKeys,
                        vector<uchar> mask = vector<uchar>(0, 0));


int histVoteTrack(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
             vector<uchar>& mask, vector<uchar>* prevMask=NULL);

int histVoteMatch(const vector<KeyPoint>& queryKeys, const Mat& queryDes,
             const vector<KeyPoint>& trainKeys, const Mat& trainDes,
             vector<DMatch> &matches, vector<uchar>& mask);

int histVoteUV(const vector<Point2f>& uvs, vector<uchar>& mask, Point2f& uvmid);

// skew-symmetric mapping
void sk_sym(Mat &vec, Mat &hat);

void triangulate(const KeyPoint &kp1, const KeyPoint &kp2, const Mat &P1, const Mat &P2, Mat &x3D);
void triangulate(const Point2f &pt1, const Point2f &pt2, const Mat &P1, const Mat &P2, Mat &x3D);
Mat getPrjctnMtrx(Mat K, Mat wTc);

void prjcPts2Cam(Mat P, const vector<Point3f>& pt3ds, vector<Point2f>& pt2fs);
Point2f prjcPt2Cam(Mat P, const Point3f& pt3d);
Point2f prjcPt2Cam(const Mat& K, const Mat &T, const Point3f& pt3);

bool checkParallax(const Point3f& o1, const Point3f& o2, const Point3f& pt3);

bool depthAccepted(const Mat& K, const Mat& T, const Point3f& pt, float min, float max);

float getDepth(const Mat& T, const Point3f& pt);

Point3f XYZfromUVDepth(const Point2f& uv, float depth, const Mat& K);
Point3f XYZfromUVLength(const Point2f& uv, float length, const Mat &K);

Point3f XYZfromTransform(const Mat& T, const Point3f& pt3);

Point2i argmax2d(const Mat& hist);
Mat histogram2d(const vector<float>& x, const vector<float>& y,
                const vector<float>& xbins, const vector<float>& ybins);

// `histogram3d' Not tested. Do not directly use it.
Mat histogram3d(const vector<float>& x, const vector<float>& y, const vector<float>& z,
                  const vector<float>& xbins, const vector<float>& ybins, const vector<float>& zbins);
// `argmax3d` not tested, do not directly use it
Point3i argmax3d(const Mat& hist);

Point3f getXYZsigma(const Point3f& measureXYZ, const Point2f& measureUV, const Mat &K);

} // namespace scv

#endif
