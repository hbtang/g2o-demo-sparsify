// sugarCV: tool functions for opencv
// By ZHEGN Fan fzheng@link.cuhk.edu.hk

// Use namespace: scv (Sugar CV)

#include "sugarCV.h"
#include "sugarVec.h"

namespace scv{

using namespace svec;

Mat inv(const Mat &T4x4){
    assert(T4x4.cols == 4 && T4x4.rows == 4);
    Mat RT = T4x4.rowRange(0,3).colRange(0,3).t();
    Mat t = -RT * T4x4.rowRange(0,3).col(3);
    Mat T = Mat::eye(4,4,CV_32FC1);
    RT.copyTo(T.rowRange(0,3).colRange(0,3));
    t.copyTo(T.rowRange(0,3).col(3));
    return T;
}

void pts2Ftrs(const vector<KeyPoint>& _orgnFtrs, const vector<Point2f>& _points, vector<KeyPoint>& _features) {
    _features.resize(_points.size());
    for (size_t i = 0; i < _points.size(); i ++) {
        _features[i] = _orgnFtrs[i];
        _features[i].pt = _points[i];
    }
}

Mat drawKeys(const vector<KeyPoint> keys, const Mat &img,
             vector<uchar> mask) {
    if (mask.size() == 0) {
        mask = vector<uchar>(keys.size(), true);
    }
    Mat out = img.clone();
    if (img.channels() == 1)
        cvtColor(img, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < mask.size(); i++) {
        if (!mask[i])
            continue;
        Point2f pt1 = keys[i].pt;
        circle(out, pt1, 3, Scalar(0, 0, 200), 2);
    }
    return out.clone();
}

Mat drawKeysWithNum(const vector<KeyPoint> keys, const Mat &img, vector<uchar> mask){
    if (mask.size() == 0) {
        mask = vector<uchar>(keys.size(), true);
    }
    Mat out = img.clone();
    if (img.channels() == 1)
        cvtColor(img, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < mask.size(); i++) {
        if (!mask[i])
            continue;
        Point2f pt1 = keys[i].pt;
        circle(out, pt1, 3, Scalar(0, 0, 200), 2);
        putText(out,to_string(i), pt1+Point2f(2,-2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    }
    return out.clone();
}

Mat drawQueryKeys(const vector<KeyPoint> keys, const Mat &img,
                  const vector<DMatch>& matches,
                  vector<uchar> mask) {
    if (mask.size() == 0) {
        mask = vector<uchar>(keys.size(), true);
    }
    Mat out = img.clone();
    if (img.channels() == 1)
        cvtColor(img, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < mask.size(); i++) {
        if (!mask[i])
            continue;
        Point2f pt1 = keys[matches[i].queryIdx].pt;
        circle(out, pt1, 3, Scalar(0, 0, 200), 2);
    }
    return out.clone();
}

Mat drawMatchesInOneImg(const vector<KeyPoint> queryKeys,
                        const Mat &trainImg, const vector<KeyPoint> trainKeys,
                        const vector<DMatch> &matches, vector<uchar> mask) {
    if (mask.size() == 0) {
        mask = vector<uchar>(matches.size(), true);
    }
    Mat out = trainImg.clone();
    if (trainImg.channels() == 1)
        cvtColor(trainImg, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < mask.size(); i++) {
        if (!mask[i])
            continue;
        Point2f pt1 = trainKeys[matches[i].trainIdx].pt;
        Point2f pt2 = queryKeys[matches[i].queryIdx].pt;
        circle(out, pt1, 2, Scalar(0, 200, 0), 2);
        circle(out, pt2, 2, Scalar(0, 0, 200), 2);
        line(out, pt1, pt2, Scalar(200, 0, 0));
    }
    return out.clone();
}

Mat drawMatchesInOneImg(const vector<KeyPoint> queryKeys,
                        const Mat &trainImg, const vector<KeyPoint> trainKeys,
                        vector<uchar> mask) {
    if (mask.size() == 0) {
        mask = vector<uchar>(queryKeys.size(), true);
    }
    Mat out = trainImg.clone();
    if (trainImg.channels() == 1)
        cvtColor(trainImg, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < mask.size(); i++) {
        if (!mask[i])
            continue;
        Point2f pt1 = trainKeys[i].pt;
        Point2f pt2 = queryKeys[i].pt;
        circle(out, pt1, 2, Scalar(0, 200, 0), 2);
        circle(out, pt2, 2, Scalar(0, 0, 200), 2);
        line(out, pt1, pt2, Scalar(200, 0, 0));
    }
    return out.clone();
}


Point2i argmax2d(const Mat &hist){
    double min, max;
    Point2i minLoc, maxLoc;
    cv::minMaxLoc(hist, &min, &max, &minLoc, &maxLoc);
    return maxLoc;
}

Mat histogram2d(const vector<float> &x, const vector<float> &y, const vector<float> &xbins, const vector<float> &ybins){
    assert(xbins.size() > 1 && ybins.size() > 1);
    assert(x.size() == y.size());
    Mat mtrx(ybins.size()-1, xbins.size()-1, CV_16SC1, Scalar(0));
    for (unsigned i = 0; i < x.size(); i++){
        unsigned xLoc = xbins.size(), yLoc = ybins.size();
        for (unsigned j = 1; j < xbins.size(); j++){
            if (x[i] < xbins[j] && x[i] >= xbins[j - 1]){
                xLoc = j - 1;
                break;
            }
        }
        if(x[i]==xbins.back()){
            xLoc = xbins.size()-2;
        }
        for (unsigned j = 1; j < ybins.size(); j++){
            if (y[i] < ybins[j] && y[i] >= ybins[j - 1]){
                yLoc = j - 1;
                break;
            }
        }
        if(y[i]==ybins.back()){
            yLoc = ybins.size()-2;
        }
        if (xLoc < xbins.size() - 1 && yLoc < ybins.size() - 1){
            //mtrx[yLoc][xLoc]++;
            mtrx.at<short>(yLoc, xLoc) += 1;
        }

    }
    assert(!(cv::sum(mtrx)[0] - x.size()));
    return mtrx;
}


int histVoteTrack(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
             vector<uchar>& mask, vector<uchar> *prevMask){
    assert(kp1.size() == kp2.size());
    assert(kp1.size() <= mask.size());
    if(prevMask) assert(kp1.size() <= prevMask->size());

    size_t num = kp1.size();
    vector<float> *x = new vector<float>(0);
    vector<float> *y = new vector<float>(0);
    (*x).reserve(num); (*y).reserve(num);
    for (unsigned i = 0; i<num; i++){
        if(mask[i]){
            (*x).push_back(kp2[i].pt.x - kp1[i].pt.x);
            (*y).push_back(kp2[i].pt.y - kp1[i].pt.y);
        }
        /*
        if(fabs(x[i])>20 || fabs(y[i])>20){
            x[i] = 0;
            y[i] = 0;
            mask[i] = 0;
        }
        */
    }

    // create histogram
    float xMid, yMid;
    num = (*x).size();
    if (num > 2){
        vector<float> xedges = arange(min((*x)), max((*x)), BW);
        vector<float> yedges = arange(min((*y)), max((*y)), BW);
        Mat hist = histogram2d((*x), (*y), xedges, yedges);
        Point2i xyId = scv::argmax2d(hist);
        xMid = (xedges[xyId.x] + xedges[xyId.x]+BW) / 2.f;
        yMid = (yedges[xyId.y] + yedges[xyId.y]+BW) / 2.f;
    }
    else{
        xMid = median((*x));
        yMid = median((*y));
    }

    // mean shift to find the accurate mode
    vector<float> *x_samp = new vector<float>(0);
    vector<float> *y_samp = new vector<float>(0);
    unsigned count;
    for (int i = 0; i<1000; i++) {        
        float msx(0), msy(0);
        count = 0;
        (*x_samp).clear();
        (*y_samp).clear();
        for (unsigned j = 0; j<num; j++) {
            float dis = sqrt(pow((*x)[j] - xMid, 2) + pow((*y)[j] - yMid, 2));
            if (dis <= L) {
                msx += (*x)[j];
                msy += (*y)[j];
                (*x_samp).push_back((*x)[j]);
                (*y_samp).push_back((*y)[j]);
                count++;
            }
        }
        if (count>0){
            msx = msx / (float)count;
            msy = msy / (float)count;
        }
        if (fabs(msx - xMid) <= 0.000001 &&
                fabs(msy - yMid) <= 0.000001) {
            break;
        }
        else {
            xMid = msx;
            yMid = msy;
        }
    }

    delete x;
    delete y;

    // Use the samples within the radius to perform parameter estimation of Laplacian distribution
    float x_sampMid = median((*x_samp));
    float y_sampMid = median((*y_samp));
    float x_s_b(0), y_s_b(0);
    for (unsigned i = 0; i<count; i++) {
        x_s_b += fabs((*x_samp)[i] - x_sampMid);
        y_s_b += fabs((*y_samp)[i] - y_sampMid);
    }
    x_s_b = x_s_b / (float)count;
    y_s_b = y_s_b / (float)count;
    x_s_b = x_s_b>0.3f ? x_s_b : 0.3f;
    y_s_b = y_s_b>0.3f ? y_s_b : 0.3f;
    float x_maxbdy = -log(2.f*(1.f - INRATIO))*x_s_b + x_sampMid;
    float y_maxbdy = -log(2.f*(1.f - INRATIO))*y_s_b + y_sampMid;
    float x_minbdy = 2.f*x_sampMid - x_maxbdy;
    float y_minbdy = 2.f*y_sampMid - y_maxbdy;

    delete x_samp;
    delete y_samp;

    // determine inliers
    num = kp1.size();
    int nInliers = 0;
    for (unsigned i = 0; i<num; i++) {
        float dx = -kp1[i].pt.x+kp2[i].pt.x;
        float dy = -kp1[i].pt.y+kp2[i].pt.y;
        if (dx<x_maxbdy && dx>x_minbdy &&
                dy<y_maxbdy && dy>y_minbdy) {
            bool prevTrue = true;
            if(prevMask) prevTrue = (*prevMask)[i];
            if(mask[i] && prevTrue)  nInliers++;
        }  else {
            mask[i] = false;
            if(prevMask) (*prevMask)[i] = false;
        }
    }
    return nInliers;
}

int histVoteMatch(const vector<KeyPoint> &queryKeys, const Mat& queryDes,
             const vector<KeyPoint> &trainKeys, const Mat& trainDes,
             vector<DMatch>& matches, vector<uchar> &mask){
    assert(queryKeys.size()!=0);
    BFMatcher matcher;
    vector< vector<DMatch> > *vecMatches = new vector< vector<DMatch> >;
    matcher.knnMatch(queryDes, trainDes, (*vecMatches), 2);

    // ratio test
    matches.clear();
    vector<float> *x = new vector<float>(0);
    vector<float> *y = new vector<float>(0);
    unsigned sz = (*vecMatches).size();
    for (unsigned i = 0; i<sz; i++) {
        DMatch mtch0 = (*vecMatches)[i][0];
        DMatch mtch1 = (*vecMatches)[i][1];
        if (mtch0.distance < dr*mtch1.distance) {
            matches.push_back(mtch0);
            (*x).push_back(trainKeys[mtch0.trainIdx].pt.x - queryKeys[mtch0.queryIdx].pt.x);
            (*y).push_back(trainKeys[mtch0.trainIdx].pt.y - queryKeys[mtch0.queryIdx].pt.y);
        }
    }

    delete vecMatches;

    unsigned num = (*x).size();
    if(!num){
        matches.clear();
        mask.clear();
        return -1;
    }
    mask.resize(num);

    // create histogram
    float xMid, yMid;
    //vector<unsigned> xyId;
    if (num > 2){
        vector<float> xedges = arange(min((*x)), max((*x)), BW);
        vector<float> yedges = arange(min((*y)), max((*y)), BW);
        Mat hist = histogram2d((*x), (*y), xedges, yedges);
        Point2i xyId = scv::argmax2d(hist);
        xMid = (xedges[xyId.x] + xedges[xyId.x]+BW) / 2.f;
        yMid = (yedges[xyId.y] + yedges[xyId.y]+BW) / 2.f;
    }
    else{
        xMid = median((*x));
        yMid = median((*y));
    }

    // mean shift to find the accurate mode
    vector<float> *x_samp = new vector<float>(0);
    vector<float> *y_samp = new vector<float>(0);
    unsigned count;
    for (int i = 0; i<1000; i++) {
        float msx(0), msy(0);
        count = 0;
        (*x_samp).clear();
        (*y_samp).clear();
        for (unsigned j = 0; j<num; j++) {
            float dis = sqrt(pow((*x)[j] - xMid, 2) + pow((*y)[j] - yMid, 2));
            if (dis <= L) {
                msx += (*x)[j];
                msy += (*y)[j];
                (*x_samp).push_back((*x)[j]);
                (*y_samp).push_back((*y)[j]);
                count++;
            }
        }
        if (count>0){
            msx = msx / (float)count;
            msy = msy / (float)count;
        }
        if (fabs(msx - xMid) <= 0.000001 &&
                fabs(msy - yMid) <= 0.000001) {
            break;
        }
        else {
            xMid = msx;
            yMid = msy;
        }
    }


    // Use the samples within the radius to perform parameter estimation of Laplacian distribution
    float x_sampMid = median((*x_samp));
    float y_sampMid = median((*y_samp));
    float x_s_b(0), y_s_b = (0);
    for (unsigned i = 0; i<count; i++) {
        x_s_b += fabs((*x_samp)[i] - x_sampMid);
        y_s_b += fabs((*y_samp)[i] - y_sampMid);
    }
    x_s_b = x_s_b / (float)count;
    y_s_b = y_s_b / (float)count;
    x_s_b = x_s_b>0.3f ? x_s_b : 0.3f;
    y_s_b = y_s_b>0.3f ? y_s_b : 0.3f;
    float x_maxbdy = -log(2.f*(1.f - INRATIO))*x_s_b + x_sampMid;
    float y_maxbdy = -log(2.f*(1.f - INRATIO))*y_s_b + y_sampMid;
    float x_minbdy = 2.f*x_sampMid - x_maxbdy;
    float y_minbdy = 2.f*y_sampMid - y_maxbdy;

    delete x_samp;
    delete y_samp;

    // determine inliers
    num = mask.size();
    int nInliers = 0;
    for (unsigned i = 0; i<num; i++) {
        if ((*x)[i]<x_maxbdy && (*x)[i]>x_minbdy &&
                (*y)[i]<y_maxbdy && (*y)[i]>y_minbdy) {
            mask[i] = true;
            nInliers++;
        }
        else {
            mask[i] = false;
        }
    }
    delete x;
    delete y;
    return nInliers;
}


int histVoteUV(const vector<Point2f> &uvs, vector<uchar> &mask, Point2f &uvmid){

    assert(uvs.size() <= mask.size());
    unsigned num = uvs.size();
    vector<float> *x = new vector<float>(0);
    vector<float> *y = new vector<float>(0);
    (*x).reserve(num); (*y).reserve(num);
    for(unsigned i=0; i<num; i++){
        if(mask[i]){
            (*x).push_back(uvs[i].x);
            (*y).push_back(uvs[i].y);
        }
    }

    // create histogram
    float xMid, yMid;
    const float bw = 3.f;
    num = (*x).size();
    if (num > 2){
        vector<float> xedges = arange(min((*x)), max((*x)), bw);
        vector<float> yedges = arange(min((*y)), max((*y)), bw);
        Mat hist = histogram2d((*x), (*y), xedges, yedges);
        Point2i xyId = scv::argmax2d(hist);
        xMid = (xedges[xyId.x] + xedges[xyId.x]+bw) / 2.f;
        yMid = (yedges[xyId.y] + yedges[xyId.y]+bw) / 2.f;
    }
    else{
        xMid = median((*x));
        yMid = median((*y));
    }


    // mean shift to find the accurate mode
    vector<float> *x_samp = new vector<float>(0);
    vector<float> *y_samp = new vector<float>(0);
    unsigned count;
    const float radius2 = 15.f;
    for (int i = 0; i<1000; i++) {
        float msx(0), msy(0);
        count = 0;
        (*x_samp).clear();
        (*y_samp).clear();
        for (unsigned j = 0; j<num; j++) {
            float dis2 = pow((*x)[j] - xMid, 2) + pow((*y)[j] - yMid, 2);
            if (dis2 <= radius2) {
                msx += (*x)[j];
                msy += (*y)[j];
                (*x_samp).push_back((*x)[j]);
                (*y_samp).push_back((*y)[j]);
                count++;
            }
        }
        if (count>0){
            msx = msx / (float)count;
            msy = msy / (float)count;
        }
        if (fabs(msx - xMid) <= 0.000001 &&
                fabs(msy - yMid) <= 0.000001) {
            break;
        }
        else {
            xMid = msx;
            yMid = msy;
        }
    }
    delete x;
    delete y;

    // Use the samples within the radius to perform parameter estimation of Laplacian distribution
    const float inratio = INRATIO;
    float x_sampMid = median((*x_samp));
    float y_sampMid = median((*y_samp));
    float x_s_b(0), y_s_b(0);
    for (unsigned i = 0; i<count; i++) {
        x_s_b += fabs((*x_samp)[i] - x_sampMid);
        y_s_b += fabs((*y_samp)[i] - y_sampMid);
    }
    x_s_b = x_s_b / (float)count;
    y_s_b = y_s_b / (float)count;
    x_s_b = x_s_b>0.3f ? x_s_b : 0.3f;
    y_s_b = y_s_b>0.3f ? y_s_b : 0.3f;
    float x_maxbdy = -log(2.f*(1.f - inratio))*x_s_b + x_sampMid;
    float y_maxbdy = -log(2.f*(1.f - inratio))*y_s_b + y_sampMid;
    float x_minbdy = 2.f*x_sampMid - x_maxbdy;
    float y_minbdy = 2.f*y_sampMid - y_maxbdy;

    delete x_samp;
    delete y_samp;


    // determine inliers
    num = uvs.size();
    int nInliers = 0;
    for (unsigned i = 0; i<num; i++) {
        if (uvs[i].x<x_maxbdy && uvs[i].x>x_minbdy &&
                uvs[i].y<y_maxbdy && uvs[i].y>y_minbdy) {
            if(mask[i]) nInliers++;
        }
        else {
            mask[i] = 0;
        }
    }
    uvmid.x = xMid;
    uvmid.y = yMid;
    return nInliers;
}

void sk_sym(Mat &vec, Mat &hat){
    Mat mat(3,3,CV_32FC1, Scalar(0));
    mat.at<float>(0,1) = -vec.at<float>(2,0);
    mat.at<float>(0,2) = vec.at<float>(1,0);
    mat.at<float>(1,0) = vec.at<float>(2,0);
    mat.at<float>(1,2) = -vec.at<float>(0,0);
    mat.at<float>(2,0) = -vec.at<float>(1,0);
    mat.at<float>(2,1) = vec.at<float>(0,0);
    cv::swap(mat, hat);

}


void triangulate(const KeyPoint &kp1, const KeyPoint &kp2, const Mat &P1, const Mat &P2, Mat &x3D){
    Mat A(4,4,CV_32FC1);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    Mat u,w,vt;
    SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A|SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void triangulate(const Point2f &pt1, const Point2f &pt2, const Mat &P1, const Mat &P2, Mat &x3D){
    Mat A(4,4,CV_32FC1);

    A.row(0) = pt1.x*P1.row(2)-P1.row(0);
    A.row(1) = pt1.y*P1.row(2)-P1.row(1);
    A.row(2) = pt2.x*P2.row(2)-P2.row(0);
    A.row(3) = pt2.y*P2.row(2)-P2.row(1);

    Mat u, w, vt;
    SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}


Mat getPrjctnMtrx(Mat K, Mat wTc){
    // K 3x3: camera matrix
    // wTc 4x4: homogeneous transformation matrix, camera wrt world
    Mat Tinv = inv(wTc);
    return K*Tinv.rowRange(0,3);
}


void prjcPts2Cam(Mat P, const vector<Point3f> &pt3ds, vector<Point2f>& pt2fs){
    pt2fs.resize(pt3ds.size());
    for(size_t i=0; i<pt2fs.size(); i++){        
        pt2fs[i] = prjcPt2Cam(P, pt3ds[i]);
    }
}

Point2f prjcPt2Cam(Mat P, const Point3f& pt3d){
    Mat_<float> dst(3,1);
    Mat src = (Mat_<float>(4,1) << pt3d.x,pt3d.y,pt3d.z,1.f);
    dst = P*src;
    float x = dst.at<float>(0,0);
    float y = dst.at<float>(1,0);
    float z = dst.at<float>(2,0);
    return Point2f(x/z, y/z);
}

Point2f prjcPt2Cam(const Mat &K, const Mat &T, const Point3f &pt3){
    Mat R = T.rowRange(0,3).colRange(0,3);
    Mat t = T.rowRange(0,3).col(3);
    Mat xyz = (Mat_<float>(3,1) << pt3.x, pt3.y, pt3.z);
    Mat pt3out = R*xyz + t;
    Mat xyw = K * pt3out;
    return Point2f(xyw.at<float>(0,0)/xyw.at<float>(2,0),
                   xyw.at<float>(1,0)/xyw.at<float>(2,0));

}


bool checkParallax(const Point3f &o1, const Point3f &o2, const Point3f &pt3){
    Point3f p1 = pt3 - o1;
    Point3f p2 = pt3 - o2;
    float cosParallax = cv::norm(p1.dot(p2)) / ( cv::norm(p1) * cv::norm(p2) );
    return cosParallax < 0.9998f;
}


bool depthAccepted(const Mat &K, const Mat &T, const Point3f &pt, float min, float max){
    Mat pth = (Mat_<float>(4,1) << pt.x,pt.y,pt.z,1);
    Mat ptCam = T * pth;
    return (ptCam.at<float>(2,0) >= min && ptCam.at<float>(2,0) <= max);
}

float getDepth(const Mat &T, const Point3f &pt){
    Mat pth = (Mat_<float>(4,1) << pt.x,pt.y,pt.z,1);
    Mat ptCam = inv(T)*pth;
    return ptCam.at<float>(2,0);
}

Point3f XYZfromUVDepth(const Point2f &uv, float depth, const Mat& K){
    float fx = K.at<float>(0,0);
    float fy = K.at<float>(1,1);
    float cx = K.at<float>(0,2);
    float cy = K.at<float>(1,2);
    float x = (uv.x-cx)*depth/fx;
    float y = (uv.y-cy)*depth/fy;
    return Point3f(x,y,depth);
}

Point3f XYZfromUVLength(const Point2f &uv, float length, const Mat &K){
    Point3f xyz_ = XYZfromUVDepth(uv, length, K);
    float length_ = sqrt(xyz_.x*xyz_.x + xyz_.y*xyz_.y + xyz_.z*xyz_.z);
    float r = length / length_;
    return xyz_*r;
}

Point3f XYZfromTransform(const Mat &T, const Point3f &pt3){
    Mat R = T.rowRange(0,3).colRange(0,3);
    Mat t = T.rowRange(0,3).col(3);
    Mat xyz = (Mat_<float>(3,1) << pt3.x, pt3.y, pt3.z);
    Mat pt3out = R*xyz + t;
    return Point3f(pt3out);
}


// `histogram3d' Not tested. Do not directly use it.
Mat histogram3d(const vector<float>& x, const vector<float>& y, const vector<float>& z,
                  const vector<float>& xbins, const vector<float>& ybins, const vector<float>& zbins){

    assert(x.size() == y.size() && y.size() == z.size());
    assert(xbins.size() > 1 && ybins.size() >1 && zbins.size() > 1);

    int xsz = xbins.size();
    int ysz = ybins.size();
    int zsz = zbins.size();

    int sizes[] = {xsz-1, ysz-1, zsz-1};
    Mat mtrx(3, sizes, CV_16U, Scalar(0));

    for(unsigned i = 0; i < x.size(); i++){
        unsigned xLoc = xbins.size();
        unsigned yLoc = ybins.size();
        unsigned zLoc = zbins.size();
        for(unsigned j = 1; j < xbins.size(); j++){
            if (x[i] < xbins[j] && x[i] >= xbins[j - 1]){
                xLoc = j - 1;
                break;
            }
        }
        if(x[i] == xbins.back()){
            xLoc = xbins.size() - 2;
        }

        for(unsigned j = 1; j < ybins.size(); j++){
            if (y[i] < ybins[j] && y[i] >= ybins[j - 1]){
                yLoc = j - 1;
                break;
            }
        }
        if(y[i] == ybins.back()){
            yLoc = ybins.size() - 2;
        }

        for(unsigned j = 1; j < zbins.size(); j++){
            if (z[i] < zbins[j] && z[i] >= zbins[j - 1]){
                zLoc = j - 1;
                break;
            }
        }
        if(z[i] == zbins.back()){
            zLoc = zbins.size() - 2;
        }

        if( xLoc<xbins.size()-1 &&
                yLoc<ybins.size()-1 &&
                zLoc<zbins.size()-1){
            mtrx.at<unsigned>(xLoc, yLoc, zLoc)++;
        }
    }

    return mtrx;

}

// `argmax3d` not tested, do not directly use it
Point3i argmax3d(const Mat &hist){
    int xsz = hist.size[0];
    int ysz = hist.size[1];
    int zsz = hist.size[2];


    int xId(0), yId(0), zId(0);
    unsigned max = 0;
    unsigned temp;

    for(int i=0; i<xsz; i++){
        for(int j=0; j<ysz; j++){
            for(int k=0; k<zsz; k++){
                temp = hist.at<unsigned>(i, j, k);
                if(temp > max){
                    max = temp;
                    xId = i; yId = j; zId = k;
                }
            }
        }
    }

    return Point3i(xId, yId, zId);
}

Point3f getXYZsigma(const Point3f &measureXYZ, const Point2f &measureUV, const Mat& K){
    float length = sqrt(measureXYZ.x * measureXYZ.x + measureXYZ.y * measureXYZ.y + measureXYZ.z * measureXYZ.z);
    Point3f xyz = XYZfromUVLength(measureUV, length, K);
    Point3f err = xyz - measureXYZ;
    return Point3f(fabs(err.x), fabs(err.y), fabs(err.z));
}

} // namespace cvtool
