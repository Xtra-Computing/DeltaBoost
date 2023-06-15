//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDTPARAM_H
#define FEDTREE_GBDTPARAM_H

#include <string>
#include <FedTree/common.h>

// Todo: gbdt params, refer to ThunderGBM https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
struct GBDTParam {
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    float column_sampling_rate;
    std::string path;
    std::string test_path;
    int verbose;
    bool profiling;
    bool bagging;
    int n_parallel_trees;
    float learning_rate;
    std::string objective;
    int num_class;
    int tree_per_round; // #tree of each round, depends on #class

    //for histogram
    int max_num_bin;

    int n_device;

    std::string tree_method;
    std::string metric;

    std::string remain_data_path;
    std::string delete_data_path;
    std::string save_model_name;

    bool reorder_label = false;
};

struct DeltaBoostParam : public GBDTParam {
    bool enable_delta = false;
    std::string dataset_name;
    float_type remove_ratio = 0.0;
    float_type min_diff_gain = 0;
    float_type max_range_gain = 0;
    int n_used_trees = 0;
    int max_bin_size = 100;
    float_type gain_alpha = 0.0;
    int nbr_size = 1;
    float_type delta_gain_eps_feature = 0.0; // eps for features
    float_type delta_gain_eps_sn = 0.0;  // eps for split neighbors in a feature
    int hash_sampling_round = 1;  // Round of sampling (by hash). After <hash_sampling_round>, all the instances will be trained once.
                                         // sampling_ratio = 1 / hash_sampling_round
    bool perform_remove = true;
    int n_quantize_bins = 0;    // 0 means no quantization, >1 means quantization with <n_quantize_bins> bins
    size_t seed = 0;
    float_type min_gain = 0.0;  // minimum gain allow to split

    DeltaBoostParam() = default;

    DeltaBoostParam(const GBDTParam *gbdt_param, const DeltaBoostParam *deltaboost_param):
    GBDTParam(*gbdt_param), enable_delta(deltaboost_param->enable_delta),
    remove_ratio(deltaboost_param->remove_ratio),
    min_diff_gain(deltaboost_param->min_diff_gain),
    max_range_gain(deltaboost_param->max_range_gain),
    dataset_name(deltaboost_param->dataset_name),
    n_used_trees(deltaboost_param->n_used_trees),
    max_bin_size(deltaboost_param->max_bin_size),
    gain_alpha(deltaboost_param->gain_alpha),
    nbr_size(deltaboost_param->nbr_size),
    delta_gain_eps_feature(deltaboost_param->delta_gain_eps_feature),
    delta_gain_eps_sn(deltaboost_param->delta_gain_eps_sn),
    hash_sampling_round(deltaboost_param->hash_sampling_round),
    perform_remove(deltaboost_param->perform_remove),
    n_quantize_bins(deltaboost_param->n_quantize_bins),
    seed(deltaboost_param->seed),
    min_gain(deltaboost_param->min_gain) {
        if (deltaboost_param->n_used_trees > 0) {
            this->n_used_trees = deltaboost_param->n_used_trees;
        } else {
            this->n_used_trees = gbdt_param->n_trees;
        }
    }
};

#endif //FEDTREE_GBDTPARAM_H
