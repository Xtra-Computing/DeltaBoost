//
// Created by liqinbin on 10/14/20.
// ThunderGBM common.cpp: https://github.com/Xtra-Computing/thundergbm/blob/master/src/thundergbm/util/common.cpp
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#include "FedTree/common.h"
INITIALIZE_EASYLOGGINGPP

//#define EPSILON (std::numeric_limits<float_type>::epsilon())


std::ostream &operator<<(std::ostream &os, const int_float &rhs) {
    os << string_format("%d/%f", thrust::get<0>(rhs), thrust::get<1>(rhs));
    return os;
}



bool ft_eq(float_type a, float_type b, const float_type eps) {
    return ((a - b) < eps) && ((b - a) < eps);
}

bool ft_ge(float_type a, float_type b, const float_type eps) {
    return a > b - eps;
}

bool ft_le(float_type a, float_type b, const float_type eps) {
    return a < b + eps;
}

std::vector<bool> indices_to_hash_table(const std::vector<int> &indices, size_t size) {
    /**
     * Map a vector of indices to a hash table of size <size>. All the indices that exist in <indices> are set to true
     * in the returned vector; otherwise, are set to false.
     */
    std::vector<bool> hash_table(size, false);
//#pragma omp parallel for  // wield bug
    for (int i = 0; i < indices.size(); ++i) {
        hash_table[indices[i]] = true;
    }
    return hash_table;
}

void clean_gh_(vector<GHPair>& ghs) {
    ghs.erase(std::remove_if(ghs.begin(), ghs.end(), [](GHPair &gh) {
        return gh.encrypted;
    }), ghs.end());
};
void clean_indices_(vector<int>& indices) {
    indices.erase(std::remove_if(indices.begin(), indices.end(), [](int i) {
        return i == -1;
    }), indices.end());
};

void clean_indices_(vector<float_type>& indices) {
    indices.erase(std::remove_if(indices.begin(), indices.end(), [](int i) {
        return i < 0;
    }), indices.end());
};