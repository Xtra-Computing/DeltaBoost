//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_CUT_H
#define FEDTREE_HIST_CUT_H

#include "FedTree/common.h"
#include "FedTree/dataset.h"
#include "FedTree/MurmurHash3.h"
#include "openssl/md5.h"
#include <random>
#include <utility>

class HistCut {
public:

    // The vales of cut points
    SyncArray<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    SyncArray<int> cut_col_ptr;
    // The feature id for current cut point
    SyncArray<int> cut_fid;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points_val.copy_from(cut.cut_points_val);
        cut_col_ptr.copy_from(cut.cut_col_ptr);
    }

    // equally divide the feature range to get cut points
    // void get_cut_points(float_type feature_min, float_type feature_max, int max_num_bins, int n_instances);
    void get_cut_points_by_data_range(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_fast(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_by_n_instance(DataSet &dataset, int max_num_bins);
    void get_cut_points_by_feature_range(vector<vector<float_type>> f_range, int max_num_bins);
    void get_cut_points_by_parties_cut_sampling(vector<HistCut> &parties_cut, int max_num_bin);
};

class RobustHistCut {
public:
    // The values of cut points C1, C2, ..., Cn sorted in descending order.
    // The bins look like  (+inf, C1] (C1, C2] ... (Cn-1, Cn]
    vector<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    vector<int> cut_col_ptr;
    // The feature id for current cut point
    vector<int> cut_fid;
    // Number of instances in bins B1, B2, ..., Bn. Bi is the #intances in (Ci-1, Ci]
    vector<vector<int>> n_instances_in_hist;
    // The indices in each bin. Size (n_features x n_bins_in_feature x n_instances_in_bin)
    vector<vector<vector<int>>> indices_in_hist;

    RobustHistCut() = default;

    RobustHistCut(const RobustHistCut& cut): cut_points_val(cut.cut_points_val), cut_col_ptr(cut.cut_col_ptr),
    n_instances_in_hist(cut.n_instances_in_hist) { }

    void get_cut_points_by_feature_range_balanced(DataSet &dataset, int max_bin_size, int n_instances);
    void get_cut_points_by_instance(DataSet &dataset, int max_num_bins, int n_instances);

    [[nodiscard]] inline float_type get_cut_point_val(int fid, int bid) const {
        int feature_offset = cut_col_ptr[fid];
        return cut_points_val[feature_offset + bid];
    }

    [[nodiscard]] inline auto get_cut_point_val_itr(int fid, int bid) const {
        int feature_offset = cut_col_ptr[fid];
        return cut_points_val.begin() + feature_offset + bid;
    }

    [[nodiscard]] inline float_type get_cut_point_val(int bid) const {      // global bid
        return cut_points_val[bid];
    }

    [[nodiscard]] inline auto get_cut_point_val_itr(int bid) const {       // global bid
        return cut_points_val.begin() + bid;
    }

    static bool cut_value_hash_comp(const float_type v1, const float_type v2) {
        /**
         * Compare cut values by hash. The result depends entirely on v1 and v2, but in an almost random way.
         */
        auto v1_seed = (unsigned long) std::round(v1 * 10000);
        auto v2_seed = (unsigned long) std::round(v2 * 10000);

        uint32_t v1_hash, v2_hash;
        MurmurHash3_x86_32(&v1_seed, sizeof(v1_seed), 0, &v1_hash);
        MurmurHash3_x86_32(&v2_seed, sizeof(v2_seed), 0, &v2_hash);

//        std::mt19937 rng_v1{v1_seed};
//        std::mt19937 rng_v2{v2_seed};
//        std::uniform_int_distribution<unsigned> dist(std::mt19937::min(), std::mt19937::max());
//        auto v1_value = dist(rng_v1);
//        auto v2_value = dist(rng_v2);
        return v1_hash < v2_hash;
    }
};


class DeltaCut: public RobustHistCut {
public:
    struct Bin {
        float_type left;
        float_type right;
        int n_instances;
        bool splittable;
        bool is_leaf = true;
        bool is_valid = true;
        int lch_id = -1;
        int rch_id = -1;
        int parent_id = -1;
        vector<int> indices;    // indices in this bin (optional, can be further optimized)

        Bin() = default;
        Bin(float_type left, float_type right, int n_instances, bool splittable, bool is_leaf=true, bool is_valid=true):
                left(left), right(right), n_instances(n_instances), splittable(splittable), is_leaf(is_leaf), is_valid(is_valid) {}
        Bin(float_type left, float_type right, int n_instances, bool splittable, bool is_leaf, bool is_valid, int lch_id, int rch_id, int parent_id):
                left(left), right(right), n_instances(n_instances), splittable(splittable), is_leaf(is_leaf), is_valid(is_valid), lch_id(lch_id), rch_id(rch_id), parent_id(parent_id) {}
        Bin(const Bin& bin) = default;

        [[nodiscard]] float_type mid_value() const {
            return (left + right) / 2;
        }
    };

    struct BinTree {
        vector<Bin> bins;

        BinTree() = default;
        BinTree(const BinTree &tree) = default;
        explicit BinTree(Bin &root) : bins({root}) {}

        [[nodiscard]] inline int get_largest_bin_id();

        void split_bin_(int bin_id, int fid, float_type split_value, const DataSet &dataset);
        void get_leaf_bins(vector<Bin> &leaf_bins) const;
        void get_split_values(vector<float_type> &split_values) const;
        void get_n_instances_in_bins(vector<int> &n_instances_in_bins) const;
        void get_indices_in_bins(vector<vector<int>> &indices_in_bins) const;
        void trim_empty_bins_();
        void remove_instances_(const vector<float_type> &values);

        void prune_(float_type threshold);
    };

    vector<BinTree> bin_trees;  // Should be the same size of n_features. Each BinTree contains the bins for one feature.

    DeltaCut() = default;
    DeltaCut(const DeltaCut &cut) = default;
    explicit DeltaCut(vector<BinTree> bin_trees): bin_trees(std::move(bin_trees)) {};

    void generate_bin_trees_(DataSet &dataset, int max_num_bins);
    void update_cut_points_(const DataSet *dataset);
    inline int n_features() const { return bin_trees.size(); }
};


#endif //FEDTREE_HIST_CUT_H
