//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_DATASET_H
#define FEDTREE_DATASET_H

#include <random>


#include "FedTree/FL/FLparam.h"
#include "common.h"
#include "syncarray.h"


class DataSet{
    public:
    ///load dataset from file
//    void load_from_file(const string& file_name, FLParam &param);
    void load_from_file(string file_name, FLParam &param);
    void load_from_csv(string file_name, FLParam &param);
//    void load_from_file_dense(string file_name, FLParam &param);
    void load_from_files(vector<string>file_names, FLParam &param);
    void load_group_file(string file_name);
    void group_label_without_reorder(int n_class);
    void group_label();

    void load_csc_from_file(string file_name, FLParam &param, int const nfeatures=500);
    void csr_to_csc();

    DataSet() = default;
    DataSet(const DataSet&) = default;

    size_t n_features() const;

    size_t n_instances() const;

//    vector<vector<float_type>> dense_mtx;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<int> group;
    vector<float_type> label;

    std::map<float_type, int> label_map;

    // csc variables
    vector<float_type> csc_val;
    vector<int> csc_row_idx;
    vector<int> csc_col_ptr;

    //Todo: SyncArray version
//    SyncArray<float_type> csr_val;
//    SyncArray<int> csr_row_ptr;
//    SyncArray<int> csr_col_idx;
//
//    SyncArray<float_type> csc_val;
//    SyncArray<int> csc_row_idx;
//    SyncArray<int> csc_col_ptr;
    // whether the dataset is to big
    bool use_cpu = true;
    bool has_csc = false;
    bool is_classification = false;
    bool has_label = true;

    vector<DataSet> sampled_datasets;
    vector<vector<int>> subset_indices;
    vector<int> row_hash;
    vector<int> original_indices;   // if this dataset is a subset, this is the original indices of the dataset, otherwise it is empty

    std::mt19937 rng;

    void get_subset(vector<int> &idx, DataSet &subset);
    DataSet& get_sampled_dataset(int cur_sampling_round);
    vector<int>& get_subset_indices(int cur_sampling_round);
    void update_sampling_by_hashing_(int total_sampling_round);
    void get_row_hash_();

    void csc_to_csr();
    void set_seed(int seed);
};

#endif //FEDTREE_DATASET_H
