//
// Created by HUSTW on 7/31/2021.
//

#include "hist_tree_builder.h"
#include <utility>

#ifndef FEDTREE_DELTA_TREE_BUILDER_H
#define FEDTREE_DELTA_TREE_BUILDER_H

typedef std::pair<int, DeltaTree::DeltaGain> gain_pair;

class DeltaTreeBuilder: public HistTreeBuilder {
public:
    void init(DataSet &dataset, const DeltaBoostParam &param, bool skip_get_bin_ids = false);

    void init_nocutpoints(DataSet &dataset, const DeltaBoostParam &param);

    void reset(DataSet &dataset, const DeltaBoostParam &param);

    vector<DeltaTree> build_delta_approximate(const SyncArray<GHPair> &gradients,
                                              std::vector<std::vector<int>>& ins2node_indices_in_tree,
                                              const vector<bool>& is_subset_indices_in_tree,
                                              bool update_y_predict = true);

    void find_split(int level) override;

    void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                      int *hist_fid, SyncArray<GHPair> &missing_gh,
                                      SyncArray<GHPair> &hist, vector<float_type> &missing_g2,
                                      vector<float_type> &hist_square);

    void compute_gain_in_a_level(vector<DeltaTree::DeltaGain> &gain, int n_nodes_in_level, int n_bins,
                                 int *hist_fid, SyncArray<GHPair> &missing_gh, vector<float_type> &hist_g2,
                                 vector<float_type> &missing_g2, SyncArray<GHPair> &hist, int n_column = 0);

    void get_split_points(vector<gain_pair> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                          SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist);
    void get_split_points(vector<DeltaTree::SplitNeighborhood> &best_split_nbr, int n_nodes_in_level, int *hist_fid,
                          SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist,
                          vector<float_type> &hist_g2,  vector<float_type> &missing_g2, int level);

    void get_topk_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                                  int n_nodes_in_level, int n_bins, int k = 1);

    int get_threshold_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                                       int n_nodes_in_level, int n_bins, float_type min_diff, float_type max_range,
                                       const vector<int> &n_samples_in_nodes);

    void update_ins2node_id() override;

    void update_ins2node_indices();

    void update_tree();

    void predict_in_training(int k);

    void get_potential_split_points(const vector<vector<gain_pair>> &candidate_idx_gain,
                                    const int n_nodes_in_level,
                                    const int *hist_fid, SyncArray<GHPair> &missing_gh,
                                    SyncArray<GHPair> &hist, vector<float_type> &hist_g2, int level);

    void get_best_split_nbr(const vector<DeltaTree::DeltaGain> &gain,
                            vector<DeltaTree::SplitNeighborhood> &best_split_nbr,
                            int n_nodes_in_level, int n_bins, int nbr_size);

    int filter_potential_idx_gain(const vector<vector<gain_pair>>& candidate_idx_gain,
                                  vector<vector<gain_pair>>& potential_idx_gain,
                                  float_type quantized_width, int max_num_potential);

    void broadcast_potential_node_indices(int node_id);

    void get_bin_ids();

    void update_random_feature_rank_(size_t seed);

    void update_random_split_nbr_rank_(size_t seed);

    void update_indices_in_split_nbr(vector<DeltaTree::SplitNeighborhood> &split_nbr, const vector<int>& node_indices);

    DeltaTree tree;
    DeltaBoostParam param;
    SyncArray<DeltaSplitPoint> sp;
//    RobustHistCut cut;
    DeltaCut cut;

    vector<int> num_nodes_per_level;    // number of nodes in each level, including potential nodes
    vector<vector<int>> ins2node_indices;   // each instance may be in multiple nodes, -1 means not in any node

    vector<int> parent_indices;     // ID: the relative index of child in the layer
                                    // Value: the relative index of its parent in the layer
    vector<int> random_feature_rank;       // random rank of feature in trees
    vector<int> random_split_nbr_rank;    // random rank of each split neighbor, the same size as #bins

    vector<bool> is_prior;       // ID: node index; Value: whether the node is prior node or not.
    float_type delta_gain_eps;   // delta_gain_eps for this tree

    float_type g_bin_width = 1.;      // bin width for gradient
    float_type h_bin_width = 1.;      // bin width for hessian

    int n_all_instances;    // number of all instances in the dataset (without subset)
};

#endif //FEDTREE_DELTA_TREE_BUILDER_H
