//
// Created by HUSTW on 7/31/2021.
//

#include "booster.h"
#include "Tree/GBDTparam.h"
#include "FedTree/Tree/delta_tree_builder.h"

#ifndef FEDTREE_DELTABOOSTER_H
#define FEDTREE_DELTABOOSTER_H

class DeltaBooster : public Booster {
public:
    void reset(DataSet &dataSet, const DeltaBoostParam &param, bool get_cut_points = true);
    void init(DataSet &dataSet, const DeltaBoostParam &delta_param, int n_all_instances, bool get_cut_points = true,
              bool skip_get_bin_ids = false);

//    void init(const GBDTParam &param, int n_instances) override;
    void boost(vector<vector<DeltaTree>>& boosted_model, vector<vector<GHPair>>& gh_pairs_per_sample,
               vector<vector<vector<int>>>& ins2node_indices_per_tree, const vector<int> &row_hash,
               const vector<bool>& is_subset_indices_in_tree);
    void boost_without_prediction(vector<vector<DeltaTree>>& boosted_model);
    vector<GHPair> quantize_gradients(const vector<GHPair> &gh, int n_bins, const vector<int> &row_hash,
                                      float_type &g_bin_width, float_type &h_bin_width);
    static float_type random_round(float_type x, float_type left, float_type right, size_t seed);
    static float_type random_round(float_type x, size_t seed);

    std::unique_ptr<DeltaTreeBuilder> fbuilder;
    DeltaBoostParam param;
    int n_all_instances = 0;
};

#endif //FEDTREE_DELTABOOSTER_H
