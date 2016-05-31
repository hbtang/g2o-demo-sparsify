// sugarVec: tool functions for std::vector
// By ZHEGN Fan fzheng@link.cuhk.edu.hk

// Use namespace: svec (Sugar Vector)

#include "sugarVec.h"

namespace svec{


int randSample(size_t _min, size_t _max, size_t _num, vector<size_t>& _sample) {
    if ((_max - _min) < _num * 3) {
        // small domain, return error
        return 1;
    }
    _sample.clear();
    vector<bool> *ifSet = new vector<bool>;
    (*ifSet).insert((*ifSet).begin(), _max-_min+1, false);
    for (size_t i = 0; i < _num; i ++) {
        size_t idx_tmp = rand() % (_max - _min + 1);
        while ((*ifSet)[idx_tmp - _min]) {
            idx_tmp = rand() % (_max - _min + 1);
        }
        (*ifSet)[idx_tmp - _min] = true;
        _sample.push_back(idx_tmp);
    }
    delete ifSet;
    return 0;
}

}
