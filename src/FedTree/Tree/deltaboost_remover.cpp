#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by zhaomin on 16/12/21.
//

#include "FedTree/Tree/deltaboost_remover.h"

[[deprecated("Prediction does not work for subsets")]]
void DeltaBoostRemover::get_info_by_prediction(const vector<vector<GHPair>> &gh_pairs_per_sample) {

    auto& trees = *trees_ptr;
    size_t n_instances = dataSet->n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    size_t num_iter = param.n_trees == -1 ? trees.size() : param.n_used_trees;
    int num_class = static_cast<int>(trees.front().size());
    std::vector<std::vector<float_type>> y_predict(n_instances, std::vector<float_type>(num_class, 0));

    //copy instances from to GPU
//    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
//    SyncArray<float_type> csr_val(dataSet->csr_val.size());
//    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
//    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
//    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
//    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

    //do prediction
    auto csr_col_idx_data = dataSet->csr_col_idx.data();
    auto csr_val_data = dataSet->csr_val.data();
    auto csr_row_ptr_data = dataSet->csr_row_ptr.data();
    auto lr = param.learning_rate;

    auto get_val = [](const int *row_idx, const float_type *row_val, int row_len, int idx,
                      bool *is_missing) -> float_type {
        //binary search to get feature value
        const int *left = row_idx;
        const int *right = row_idx + row_len;

        while (left != right) {
            const int *mid = left + (right - left) / 2;
            if (*mid == idx) {
                *is_missing = false;
                return row_val[mid - row_idx];
            }
            if (*mid > idx)
                right = mid;
            else left = mid + 1;
        }
        *is_missing = true;
        return 0;
    };

    auto get_val_dense = [](const int *row_idx, const float_type *row_val, int idx) -> float_type {
        assert(idx == row_idx[idx]);
        return row_val[idx];
    };

    auto get_next_child = [](const DeltaTree::DeltaNode& node, float_type feaValue) {
        return feaValue < node.split_value ? node.lch_index : node.rch_index;
    };

    //use sparse format and binary search
#pragma omp parallel for   // remove for debug
    for (int iid = 0; iid < n_instances; ++iid) {
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                SyncArray<float_type> y_arr(1); y_arr.host_data()[0] = dataSet->y[iid];
                SyncArray<float_type> y_pred_arr(1); y_pred_arr.host_data()[0] = sum;
                if (gh_pairs_per_sample.empty()) {
                    // no gh pairs, use the original prediction
                    SyncArray<GHPair> gh_pair_arr(1);
                    obj->get_gradient(y_arr, y_pred_arr, gh_pair_arr);
                    tree_removers[iter].gh_pairs[iid] = gh_pair_arr.host_data()[0];
                    assert(gh_pair_arr.host_data()[0].h >= 0);
                } else {
                    tree_removers[iter].gh_pairs[iid] = gh_pairs_per_sample[iter][iid];
                }

                const DeltaTree::DeltaNode *end_leaf;
                const auto &nodes = trees[iter][t].nodes;
                std::vector<int> visiting_node_indices = {0};
//                std::vector<bool> prior_flags = {true};
                while (!visiting_node_indices.empty()) {        // DFS
                    int node_id = visiting_node_indices.back();
                    visiting_node_indices.pop_back();
//                    bool is_prior = prior_flags.back();
//                    prior_flags.pop_back();
                    const auto& node = nodes[node_id];
                    tree_removers[iter].ins2node_indices[iid].push_back(node_id);

                    if (node.is_leaf) {
//                        if (is_prior) {
//                            end_leaf = &node;
//                        }
                    } else {
                        // get feature value
                        int fid = node.split_feature_id;
                        bool is_missing;
//                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                        float_type fval = get_val_dense(col_idx, row_val, fid);

//                        // potential nodes (if any)
//                        if (is_prior){
//                            for (int j = 1; j < node.potential_nodes_indices.size(); ++j) {
//                                visiting_node_indices.push_back(node.potential_nodes_indices[j]);
//                                prior_flags.push_back(false);
//                            }
//                        }

                        // prior node
                        if (!is_missing) {
                            int child_id = get_next_child(nodes[node_id], fval);
                            visiting_node_indices.push_back(child_id);
                        } else if (node.default_right) {
                            visiting_node_indices.push_back(node.rch_index);
                        } else {
                            visiting_node_indices.push_back(node.lch_index);
                        }
//                        prior_flags.push_back(is_prior);
                    }
                }
                sum += lr * end_leaf->base_weight;
            }
            y_predict[iid][t] += sum;
        }
    }


}

void DeltaBoostRemover::get_info(const vector<vector<GHPair>> &gh_pairs_per_sample,
                                 const vector<vector<vector<int>>> &ins2node_indices_per_tree) {
//    // initialize gh_pairs and ins2node_indices
//    for (int iter = 0; iter < num_iter; iter++) {
//        tree_removers[iter].gh_pairs = vector<GHPair>(n_all_instances);
//        tree_removers[iter].ins2node_indices = vector<vector<int>>(num_iter, vector<int>(n_all_instances));
//    }
    size_t num_iter = param.n_trees == -1 ? trees_ptr->size() : param.n_used_trees;
#pragma omp parallel for
    for (int iid = 0; iid < n_all_instances; ++iid) {
        for (int iter = 0; iter < num_iter; iter++) {
            tree_removers[iter].gh_pairs[iid] = gh_pairs_per_sample[iter][iid];
            tree_removers[iter].ins2node_indices[iid] = ins2node_indices_per_tree[iter][iid];
        }
    }

    auto &trees = *trees_ptr;
    // the original ins2node_indices only contains the leaf node, recursively add the parent nodes to ins2node_indices
#pragma omp parallel for
    for (int iid = 0; iid < n_all_instances; ++iid) {
        for (int iter = 0; iter < num_iter; iter++) {
            auto &ins2node_indices = tree_removers[iter].ins2node_indices[iid];
            assert(ins2node_indices.size() == 1);
            int node_id = ins2node_indices[0];
            if (node_id == -1) continue;
            while (node_id != 0) {
                node_id = trees[iter][0].nodes[node_id].parent_index;
                ins2node_indices.push_back(node_id);
            }
        }
    }

    LOG(DEBUG) << "Getting info done.";
}
