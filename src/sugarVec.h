// sugarVec: tool functions for std::vector
// By ZHEGN Fan fzheng@link.cuhk.edu.hk

// Use namespace: svec (Sugar Vector)

#ifndef SUGAR_VEC_H
#define SUGAR_VEC_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <deque>
#include <cassert>

namespace svec{
using namespace std;

typedef unsigned char uchar;

template<typename _T>
size_t count(const vector<_T>& mask, const _T& val){
    size_t no = 0;
    for(size_t i=0; i<mask.size(); i++){
        if(mask[i]==val){
            no++;
        }
    }
    return no;
}

template<typename _T>
size_t countTrue(const vector<_T>& mask){
    size_t no = 0;
    for(size_t i=0; i<mask.size(); i++){
        if(mask[i]){
            no++;
        }
    }
    return no;
}

template <typename _T>
void copy(const vector<_T>& in, vector<_T>& out, const vector<uchar>& mask){
    assert(in.size() == mask.size() && "Input and mask vector for `copy` must have same size!");
    out.resize(countTrue(mask));
    size_t j=0;
    for (size_t i = 0; i < mask.size(); i++){
        if (mask[i]){
            out[j] = in[i];
            j++;
        }
    }
}


template<typename _T>
_T max(const vector<_T>& _v){
    return *max_element(_v.begin(), _v.end());
}

template<typename _T>
size_t argmax(const vector<_T>& _v){
    return max_element(_v.begin(), _v.end()) - _v.begin();
}

template<typename _T>
_T min(const vector<_T>& _v){
    return *min_element(_v.begin(), _v.end());
}

template<typename _T>
size_t argmin(const vector<_T>& _v){
    return min_element(_v.begin(), _v.end()) - _v.begin();
}


template<typename _T>
void andand(const vector<_T> &v1, vector<_T> &vout){
    assert(v1.size() == vout.size() && "Two vectors for `andand` must have same size!");
    for(unsigned i=0; i<v1.size(); i++){
        if(vout[i]){
            vout[i] = v1[i];
        }
    }
}

template<typename _T>
double median(const vector<_T>& _v) {
    size_t sz = _v.size();
    if (sz == 0)
        return 0;
    else {
        vector<_T> vec(_v);
        std::sort(vec.begin(), vec.end());
        return sz % 2 == 0 ? (double)(vec[sz / 2] + vec[sz / 2 - 1]) / 2. : vec[sz / 2];
        }
    }

template<typename _T>
_T sum(const vector<_T>& _v){
    return accumulate(_v.begin(), _v.end(), (_T)0);
}


template<typename _T>
_T sum(const deque<_T>& _d){
    return accumulate(_d.begin(), _d.end(), (_T)0);
}


template<typename _T>
_T average(const vector<_T>& _v){
    return accumulate(_v.begin(), _v.end(), (_T)0) / (_T)_v.size();
}


template<typename _T>
_T average(const deque<_T> &_d){
    return accumulate(_d.begin(), _d.end(), (_T)0) / (_T)_d.size();
}

template<typename _T>
_T average(const vector<_T>& _v, const vector<uchar>& _mask){
    _T sum = 0.f;
    size_t num = 0;
    for(size_t i=0; i<_mask.size(); i++){
        if(_mask[i]){
            sum = sum + _v[i];
            num++;
        }
    }
    return sum/((_T)num);
}

template<typename _T>
void plus(vector<_T>& vec, const vector<_T>& add){
    assert(vec.size() == add.size() && "Two vectors for `plus` must have same size!");
    for(size_t i=0; i<vec.size(); i++){
        vec[i] += add[i];
    }
}

template<typename _T>
vector<_T> arange(const _T& begin, const _T& end, const _T& step){
    unsigned sz = (unsigned)((end - begin) / step);
    vector<_T> vec(sz+1);
    for (unsigned i = 0; i <= sz; i++){
        vec[i] = begin + (_T)i * step;
    }
    if(vec.back() - end < std::min(step*0.01, 0.0001)){
        vec.push_back(end);
    }
    return vec;
}

int randSample(size_t _min, size_t _max, size_t _num, vector<size_t>& _sample);

template<typename _T>
size_t checkDuplicate(const vector<_T>& v1, const vector<_T>& v2, vector<_T>& vout){
    vout.clear();
    for(size_t i=0; i<v1.size(); i++){
        auto it = find(v2.begin(),v2.end(),v1[i]);
        if(it!=v2.end()){
            vout.push_back(v1[i]);
        }
    }
    return vout.size();
}

template<typename _T>
vector<int> argsmax(const vector<_T>& v, int num){
    struct Comp{
        Comp( const vector<_T>& v ) : _v(v) {}
        bool operator ()(_T a, _T b) { return _v[a] > _v[b]; }
        const vector<_T>& _v;
    };

    vector<int> vx;
    vx.resize(v.size());
    for( unsigned i= 0; i<v.size(); ++i ) vx[i]= i;
    partial_sort( vx.begin(), vx.begin()+num, vx.end(), Comp(v) );
    vx.resize(num);
    return vx;
}


}// namespace svec

#endif
