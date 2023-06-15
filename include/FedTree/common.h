//
// Created by liqinbin on 10/14/20.
// ThunderGBM common.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_COMMON_H
#define FEDTREE_COMMON_H

#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT


#include "boost/serialization/vector.hpp"

#include "FedTree/util/log.h"
#include "cstdlib"
#include "config.h"
#include "thrust/tuple.h"
#include "boost/json.hpp"
#include "numeric"
#include <thrust/execution_policy.h>
//#include "FedTree/Encryption/HE.h"
#include "FedTree/Encryption/paillier.h"

namespace json = boost::json;


using std::vector;
using std::string;

#define NO_GPU \
LOG(FATAL)<<"Cannot use GPU when compiling without GPU"

//https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

//data types
typedef float float_type;

#define EPSILON 1e-9

bool ft_eq(float_type a, float_type b, float_type eps=EPSILON);

bool ft_ge(float_type a, float_type b, float_type eps=EPSILON);

bool ft_le(float_type a, float_type b, float_type eps=EPSILON);

#undef EPSILON

//CUDA macro
#ifdef USE_CUDA

#include "cuda_runtime_api.h"
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (false)

#endif

#define HOST_DEVICE __host__ __device__

struct GHPair {
    float_type g = 0.;
    float_type h = 0.;
    NTL::ZZ g_enc;
    NTL::ZZ h_enc;
    Paillier paillier;
    bool encrypted = false;

    HOST_DEVICE void homo_encrypt(const Paillier &pl) {
        if (!encrypted) {
            g_enc = pl.encrypt(NTL::to_ZZ((unsigned long) (g * 1e6)));
            h_enc = pl.encrypt(NTL::to_ZZ((unsigned long) (h * 1e6)));
            this->paillier = pl;
            g = 0;
            h = 0;
            encrypted = true;
        }
    }

    HOST_DEVICE void homo_decrypt(const Paillier &pl) {
        if (encrypted) {
            long g_dec = NTL::to_long(pl.decrypt(g_enc));
            long h_dec = NTL::to_long(pl.decrypt(h_enc));
            g = (float_type) g_dec / 1e6;
            h = (float_type) h_dec / 1e6;
            encrypted = false;
        }
    }

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g + rhs.g;
            res.h = this->h + rhs.h;
            res.encrypted = false;
        } else {
            if (!encrypted) {
                GHPair tmp_lhs = *this;
                tmp_lhs.homo_encrypt(rhs.paillier);
                res.g_enc = rhs.paillier.add(tmp_lhs.g_enc, rhs.g_enc);
                res.h_enc = rhs.paillier.add(tmp_lhs.h_enc, rhs.h_enc);
                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                GHPair tmp_rhs = rhs;
                tmp_rhs.homo_encrypt(paillier);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            } else {
                res.g_enc = paillier.add(g_enc, rhs.g_enc);
                res.h_enc = paillier.add(h_enc, rhs.h_enc);
                res.paillier = paillier;
            }
            res.encrypted = true;
        }
        return res;
    }

    HOST_DEVICE GHPair& operator+=(const GHPair &rhs) {
        *this = *this + rhs;
        return *this;
    }

    HOST_DEVICE GHPair operator-() {
        return {-g, -h, encrypted};
    }

    HOST_DEVICE GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g - rhs.g;
            res.h = this->h - rhs.h;
            res.encrypted = false;
        } else {
            GHPair tmp_lhs = *this;
            GHPair tmp_rhs = rhs;
            NTL::ZZ minus_one = NTL::to_ZZ((unsigned long) -1);
            if (!encrypted) {
                tmp_lhs.homo_encrypt(rhs.paillier);
                tmp_rhs.g_enc = rhs.paillier.mul(tmp_rhs.g_enc, minus_one);
                tmp_rhs.h_enc = rhs.paillier.mul(tmp_rhs.h_enc, minus_one);
                res.g_enc = rhs.paillier.add(tmp_lhs.g_enc, tmp_rhs.g_enc);
                res.h_enc = rhs.paillier.add(tmp_lhs.h_enc, tmp_rhs.h_enc);
                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                tmp_rhs.g *= -1;
                tmp_rhs.h *= -1;
                tmp_rhs.homo_encrypt(paillier);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            } else {
                tmp_rhs.g_enc = paillier.mul(tmp_rhs.g_enc, minus_one);
                tmp_rhs.h_enc = paillier.mul(tmp_rhs.h_enc, minus_one);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            }
            res.encrypted = true;
        }
        return res;
    }

    HOST_DEVICE bool operator==(const GHPair &rhs) const {
        return this->g == rhs.g && this->h == rhs.h;
    }

    HOST_DEVICE bool operator!=(const GHPair &rhs) const {
        return !(*this == rhs);
    }

    HOST_DEVICE GHPair() : g(0), h(0) {};

    HOST_DEVICE GHPair(float_type g, float_type h, bool encrypted) : g(g), h(h), encrypted(encrypted) {};

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {};

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {};

    GHPair(const GHPair& other) {
        g = other.g;
        h = other.h;
        g_enc = other.g_enc;
        h_enc= other.h_enc;
        paillier = other.paillier;
        encrypted = other.encrypted;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }

    HOST_DEVICE GHPair& operator=(const GHPair& other) {
        this->g = other.g;
        this->h = other.h;
        g_enc = other.g_enc;
        h_enc= other.h_enc;
        paillier = other.paillier;
        encrypted = other.encrypted;
        return *this;
    }

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar & g;
        ar & h;
        ar & encrypted;
    }

    // json parser
    friend GHPair tag_invoke(json::value_to_tag<GHPair>, json::value const& v) {
        auto &o = v.as_object();

        GHPair gh_pair;
        gh_pair.g = v.at("g").as_double();
        gh_pair.h = v.at("h").as_double();
        gh_pair.encrypted = v.at("encrypted").as_bool();

        return gh_pair;
    }

    //json parser
    friend void tag_invoke(json::value_from_tag, json::value& v, GHPair const& gh_pair) {
        v = json::object{
                {"g",         (double) gh_pair.g},
                {"h",         (double) gh_pair.h},
                {"encrypted", gh_pair.encrypted}
        };  // store all floating points as double
    }
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec) {
    std::vector<T> result;
    for (const auto & v : vec)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

std::vector<bool> indices_to_hash_table(const std::vector<int> &vec, size_t size);

template<typename T>
auto clean_vectors_(vector<vector<T>>& vectors) {
    vectors.erase(std::remove_if(vectors.begin(), vectors.end(), [](vector<T> &vec){
        return vec.empty();
    }), vectors.end());
}

template<typename T1, typename T2>
auto clean_vectors_(vector<std::map<T1, T2>>& vectors) {
    vectors.erase(std::remove_if(vectors.begin(), vectors.end(), [](auto &vec){
        return vec.empty();
    }), vectors.end());
}

template<typename T1, typename T2>
auto clean_vectors_by_indices_(vector<std::unordered_map<T1, T2>>& vectors, const vector<int> &indices) {
    vectors.erase(std::remove_if(vectors.begin(), vectors.end(), [&](auto &vec){
        int idx = &vec - &*vectors.begin();
        return indices[idx] == -1;
    }), vectors.end());
}

template<typename T, typename IteratorType>
vector<T> flatten(IteratorType itr_begin, IteratorType itr_end) {
    std::vector<T> result;
    for (auto it = itr_begin; it != itr_end; ++it)
        result.insert(result.end(), (*it).begin(), (*it).end());
    return result;
}

void clean_gh_(vector<GHPair>& ghs);
void clean_indices_(vector<int>& indices);

template<typename T>
void inclusive_scan_by_cut_points(const T *input, const int *cut_points, int n_nodes, int n_features, int n_bins, T *result) {
#pragma omp parallel for
    for (int i = 0; i < n_nodes; ++i) {
#pragma omp parallel for
        for (int j = 0; j < n_features; ++j) {
            auto start_ptr = input + i * n_bins + cut_points[j];
            auto end_ptr = input + i * n_bins + cut_points[j + 1];
            auto result_ptr = result + i * n_bins + cut_points[j];
//            thrust::inclusive_scan(thrust::host, start_ptr, end_ptr, result_ptr);
            std::partial_sum(start_ptr, end_ptr, result_ptr);
        }
    }
}

#endif //FEDTREE_COMMON_H
