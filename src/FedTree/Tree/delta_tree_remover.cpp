//
// Created by HUSTW on 8/17/2021.
//

#include <algorithm>
#include <queue>
#include <random>
#include <chrono>
#include <boost/functional/hash.hpp>
#include <unordered_set>
#include <unordered_map>

#include "FedTree/Tree/delta_tree_remover.h"

//bool DeltaTreeRemover::remove_samples_by_indices(const vector<int>& indices) {
//    /**
//     * @param id: the index of sample to be removed from the tree
//     * @return : true when a removal is successful; false when failing to remove and a retrain is needed
//     */
//
//    for (int id: indices) {
//        remove_sample_by_id(id);
//    }
//}


// for debug purpose, print element in the unordered_map
#define SHOW(X) std::cout << # X " = " << (X) << std::endl
void testPrint(std::unordered_map<int,GHPair> & m, int i )
{
    SHOW( m[i] );
    SHOW( m.find(i)->first );
}

void testPrint(std::unordered_set<std::pair<int,int>, boost::hash<std::pair<int, int>>> & m, int i, int j )
{
    SHOW( m.find({i, j}) != m.end() );
}

template class std::unordered_map<int, GHPair>;     // explicit instantiation for debug unordered_map

void DeltaTreeRemover::remove_sample_by_id(int id) {
    vector<int> indices = {id};
    vector<GHPair> gh_pair_vec = {-gh_pairs[id]};
    adjust_gradients_by_indices(indices, gh_pair_vec);
}

void DeltaTreeRemover::remove_samples_by_indices(const vector<int>& indices) {
    vector<GHPair> gh_pair_vec(indices.size());

#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
        gh_pair_vec[i] = -gh_pairs[indices[i]];
    }

    typedef std::chrono::high_resolution_clock timer;
    auto start_time = timer::now();
    auto end_time = timer::now();
    std::chrono::duration<double> duration = end_time - start_time;
//    get_invalid_sp(tree_ptr->dense_bin_id, tree_ptr->cut, indices, invalid_bids);

    start_time = timer::now();
    get_invalid_sp(tree_ptr->cut, *dataSet, indices, param.max_bin_size);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "get_invalid_sp time: " << duration.count() << "s";

    // this function is parallel
    start_time = timer::now();
    adjust_split_nbrs_by_indices(indices, gh_pair_vec, true);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "adjust_split_nbrs_by_indices time: " << duration.count() << "s";
}




[[deprecated]]
void DeltaTreeRemover::adjust_gradients_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs) {
    /**
     * @param id: the indices of sample to be adjusted gradients
     * @param delta_gh_pair: gradient and hessian to be subtracted from the tree
     */

//    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
//    SyncArray<float_type> csr_val(dataSet->csr_val.size());
//    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
//    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
//    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
//    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

    const auto csr_col_idx_data = dataSet->csr_col_idx.data();
    const auto csr_val_data = dataSet->csr_val.data();
    const auto csr_row_ptr_data = dataSet->csr_row_ptr.data();

    // update the gain of all nodes according to ins2node_indices
    vector<vector<int>> updating_node_indices(indices.size(), vector<int>(0));
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
        updating_node_indices[i] = ins2node_indices[indices[i]];
    }

    auto get_val = [&](int iid, int fid,
                   bool *is_missing) -> float_type {
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];

        //binary search to get feature value
        const int *left = col_idx;
        const int *right = col_idx + row_len;

        while (left != right) {
            const int *mid = left + (right - left) / 2;
            if (*mid == fid) {
                *is_missing = false;
                return row_val[mid - col_idx];
            }
            if (*mid > fid)
                right = mid;
            else left = mid + 1;
        }
        *is_missing = true;
        return 0;
    };

    // update GH_pair of node and parent (parallel)
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            // update sum_gh_pair
            auto &node = tree_ptr->nodes[node_id];
            #pragma omp atomic
            node.sum_gh_pair.g += delta_gh_pairs[i].g;
            #pragma omp atomic
            node.sum_gh_pair.h += delta_gh_pairs[i].h;
            #pragma omp atomic
            node.gain.self_g += delta_gh_pairs[i].g;
            #pragma omp atomic
            node.gain.self_h += delta_gh_pairs[i].h;

            // update missing_gh
            bool is_missing = false;
            float_type split_fval = get_val(indices[i], node.split_feature_id, &is_missing);
            if (is_missing) {
                #pragma omp atomic
                node.gain.missing_g += delta_gh_pairs[i].g;
                #pragma omp atomic
                node.gain.missing_h += delta_gh_pairs[i].h;
            }
        }
    }


#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];
            if (!node.is_leaf) {
                node.gain.lch_g = tree_ptr->nodes[node.lch_index].gain.self_g;
                node.gain.lch_h = tree_ptr->nodes[node.lch_index].gain.self_h;
                node.gain.rch_g = tree_ptr->nodes[node.rch_index].gain.self_g;
                node.gain.rch_h = tree_ptr->nodes[node.rch_index].gain.self_h;
            }
        }
    }

    // recalculate direction
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];

            node.calc_weight_(param.lambda, tree_ptr->g_bin_width, tree_ptr->h_bin_width);     // this lambda should be consistent with the training

            if (!node.is_leaf) {
//                node.gain.gain_value = node.default_right ? -node.gain.cal_gain_value() : node.gain.cal_gain_value();     // calculate original gain value

                // recalculate default direction
                if (node.default_right) {
                    node.gain.gain_value = -node.gain.cal_gain_value();
                    assert(node.gain.gain_value <= 0);
                    DeltaTree::DeltaGain default_left_gain(node.gain);
#pragma omp atomic
                    default_left_gain.lch_g += node.gain.missing_g;
#pragma omp atomic
                    default_left_gain.lch_h += node.gain.missing_h;
#pragma omp atomic
                    default_left_gain.rch_g -= node.gain.missing_g;
#pragma omp atomic
                    default_left_gain.rch_h -= node.gain.missing_h;
                    default_left_gain.gain_value = default_left_gain.cal_gain_value();
                    if (fabs(default_left_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        node.gain = default_left_gain;
                        node.default_right = false;
                    }
                } else {
                    node.gain.gain_value = node.gain.cal_gain_value();
                    assert(node.gain.gain_value >= 0);
                    DeltaTree::DeltaGain default_right_gain(node.gain);
#pragma omp atomic
                    default_right_gain.rch_g += node.gain.missing_g;
#pragma omp atomic
                    default_right_gain.rch_h += node.gain.missing_h;
#pragma omp atomic
                    default_right_gain.lch_g -= node.gain.missing_g;
#pragma omp atomic
                    default_right_gain.lch_h -= node.gain.missing_h;
                    default_right_gain.gain_value = -default_right_gain.cal_gain_value();
                    if (fabs(default_right_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        default_right_gain.gain_value = -default_right_gain.gain_value;
                        node.gain = default_right_gain;
                        node.default_right = true;
                    }
                }
            }
        }
    }

    sort_potential_nodes_by_gain(0);
}

[[deprecated]]
void DeltaTreeRemover::sort_potential_nodes_by_gain(int root_idx) {
    std::queue<int> processing_nodes;
    processing_nodes.push(root_idx);    // start from root node
    while(!processing_nodes.empty()) {
        int nid = processing_nodes.front();

        processing_nodes.pop();
        auto& node = tree_ptr->nodes[nid];

        if (node.is_leaf) {
            continue;
        }

        if (!node.is_valid) {
            continue;
        }

        if (!node.is_robust()) {
            // sort the nodes by descending order of gain
            std::sort(node.potential_nodes_indices.begin(), node.potential_nodes_indices.end(),
                      [&](int i, int j){
                          return fabs(tree_ptr->nodes[i].gain.gain_value) > fabs(tree_ptr->nodes[j].gain.gain_value);
                      });

            // sync the order through potential nodes
            for (int j: node.potential_nodes_indices) {
                auto &potential_node = tree_ptr->nodes[j];
                potential_node.potential_nodes_indices = node.potential_nodes_indices;
                if (!potential_node.is_leaf) {
                    processing_nodes.push(potential_node.lch_index);
                    processing_nodes.push(potential_node.rch_index);
                    if (potential_node.lch_index <= 0 || potential_node.rch_index <= 0) {
                        LOG(FATAL);
                    }
                }
            }
        } else {
            processing_nodes.push(node.lch_index);
            processing_nodes.push(node.rch_index);
            if (node.lch_index <= 0 || node.rch_index <= 0) {
                LOG(FATAL);
            }
        }
    }
}

void DeltaTreeRemover::adjust_split_nbrs_by_indices(const vector<int>& adjusted_indices, const vector<GHPair>& root_delta_gh_pairs,
                                                    bool remove_n_ins) {
    /**
     * @param id: the indices of sample to be adjusted gradients
     * @param delta_gh_pair: gradient and hessian to be subtracted from the tree
     * @param remove_n_ins: whether to remove n_instances from visited nodes
     */

//    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
//    SyncArray<float_type> csr_val(dataSet->csr_val.size());
//    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
//    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
//    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
//    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

//    const auto csr_col_idx_data = csr_col_idx.host_data();
//    const auto csr_val_data = csr_val.host_data();
//    const auto csr_row_ptr_data = csr_row_ptr.host_data();

    auto csr_col_idx_data = dataSet->csr_col_idx.data();
    auto csr_val_data = dataSet->csr_val.data();
    auto csr_row_ptr_data = dataSet->csr_row_ptr.data();

//    auto get_val = [&](int iid, int fid,
//                       bool *is_missing) -> float_type {
//        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
//        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
//        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
//
//        //binary search to get feature value
//        const int *left = col_idx;
//        const int *right = col_idx + row_len;
//
//        while (left != right) {
//            const int *mid = left + (right - left) / 2;
//            if (*mid == fid) {
//                *is_missing = false;
//                return row_val[mid - col_idx];
//            }
//            if (*mid > fid)
//                right = mid;
//            else left = mid + 1;
//        }
//        *is_missing = true;
//        return 0;
//    };

    auto get_val_dense = [&](int iid, int fid) -> float_type {
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        return row_val[fid];
    };

//    vector<vector<int>> nid_to_index_id(tree_ptr->nodes.size(), vector<int>());
//    for (int i = 0; i < indices.size(); ++i) {
//        for (auto node_id: ins2node_indices[i]) {
//            nid_to_index_id[node_id].push_back(i);
//        }
//    }

/**
 * Update gradients that are deleted
 */
    typedef std::chrono::high_resolution_clock clock;
    auto start_time = clock::now();

// update the gain of all nodes according to ins2node_indices
    vector<vector<int>> updating_node_indices(adjusted_indices.size(), vector<int>(0));
//#pragma omp parallel for      // SIGSEGV when using parallel
    for (int i = 0; i < adjusted_indices.size(); ++i) {
        updating_node_indices[i] = ins2node_indices[adjusted_indices[i]];
    }

    // update GH_pair of node and parent (parallel)
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            // update self gh
            auto &node = tree_ptr->nodes[node_id];
#pragma omp atomic
            node.sum_gh_pair.g += root_delta_gh_pairs[i].g;
#pragma omp atomic
            node.sum_gh_pair.h += root_delta_gh_pairs[i].h;
#pragma omp atomic
            node.gain.self_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
            node.gain.self_h += root_delta_gh_pairs[i].h;

            // obtain feature value of instance adjusted_indices[i] in feature node.split_feature_id
            bool is_missing = false;
//            float_type feature_val = get_val(adjusted_indices[i], node.split_feature_id, &is_missing);
            float_type feature_val = get_val_dense(adjusted_indices[i], node.split_feature_id);
            // update missing_gh
            if (is_missing) {
#pragma omp atomic
                node.gain.missing_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.missing_h += root_delta_gh_pairs[i].h;
            }

            // update left gh and right gh (can be optimized because gain will later be updated)
            if (!node.is_leaf && feature_val < node.split_value) {
#pragma omp atomic
                node.gain.lch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.lch_h += root_delta_gh_pairs[i].h;
            } else {
#pragma omp atomic
                node.gain.rch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.rch_h += root_delta_gh_pairs[i].h;
            }

            // update split neighborhood
            for (int j = 0; j < node.split_nbr.split_bids.size(); ++j) {
#pragma omp atomic
                node.split_nbr.gain[j].self_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.split_nbr.gain[j].self_h += root_delta_gh_pairs[i].h;

                if (is_missing) {
#pragma omp atomic
                    node.split_nbr.gain[j].missing_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].missing_h += root_delta_gh_pairs[i].h;
                }

                if (!node.is_leaf && feature_val < node.split_nbr.split_vals[j]) {
#pragma omp atomic
                    node.split_nbr.gain[j].lch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].lch_h += root_delta_gh_pairs[i].h;
                } else {
#pragma omp atomic
                    node.split_nbr.gain[j].rch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].rch_h += root_delta_gh_pairs[i].h;
                }

                // Remove marginal indices in each split_nbr if needed, disable invalid split values
//                if (remove_n_ins) {
//                    auto &marginal_indices = node.split_nbr.marginal_indices[j];
//                    marginal_indices.erase(adjusted_indices[i]);
//                }
            }
        }
    }
    auto end_time = clock::now();
    std::chrono::duration<float> duration = end_time - start_time;
    LOG(DEBUG) << "[Removing time] Step 1 (remove gradients) = " << duration.count();

    /**
    * Update marginal gradients that are shifted
    */
    auto overall_start_time_step2 = clock::now();

    // inference from the root, update split_nbrs layer by layer
    size_t n_nodes_in_layer;
    vector<int> visit_node_indices = {0};
    int depth = 0;

    vector<std::unordered_map<int, GHPair>> marginal_shifts(1, std::unordered_map<int, GHPair>());
    while (!visit_node_indices.empty()) {
        n_nodes_in_layer = visit_node_indices.size();
        auto start_time_level = clock::now();

        // reset the info vectors with default placeholder (for parallel)
        vector<int> next_visiting_node_indices(visit_node_indices.size() * 2, -1);
        vector<std::unordered_map<int, GHPair>> next_marginal_shifts(marginal_shifts.size() * 2, std::unordered_map<int, GHPair>());

#pragma omp parallel for
        for (int i = 0; i < n_nodes_in_layer; ++i) {
            auto start_time_in_node = clock::now();

            int node_id = visit_node_indices[i];
            auto &node = tree_ptr->nodes[node_id];

//            const auto &indices_in_node = indices_in_nodes[i];
//            const auto &marginal_indices_in_node = marginal_indices[i];
//            const auto &marginal_gh_in_node = marginal_gh[i];
            const auto &marginal_shifts_in_node = marginal_shifts[i];

            // adjust the sum_g and sum_h according to marginal indices
            auto start_time_step_2_1 = clock::now();
#pragma omp parallel for
            for (int j = 0; j < marginal_shifts_in_node.size(); ++j) {
                auto shift_it = marginal_shifts_in_node.begin();
                std::advance(shift_it, j);      // move the iterator to index j of map
                int iid = shift_it->first;
                bool is_missing = false;
//                float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);
                float_type feature_val = get_val_dense(iid, node.split_feature_id);

                // update self gh
#pragma omp atomic
                node.sum_gh_pair.g += shift_it->second.g;
#pragma omp atomic
                node.sum_gh_pair.h += shift_it->second.h;
#pragma omp atomic
                node.gain.self_g += shift_it->second.g;
#pragma omp atomic
                node.gain.self_h += shift_it->second.h;

                // update left or right gh of split_nbr based on feature_val
                for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
#pragma omp atomic
                    node.split_nbr.gain[k].self_g += shift_it->second.g;
#pragma omp atomic
                    node.split_nbr.gain[k].self_h += shift_it->second.h;

                    if (is_missing) {
#pragma omp atomic
                        node.split_nbr.gain[k].missing_g += shift_it->second.g;
#pragma omp atomic
                        node.split_nbr.gain[k].missing_h += shift_it->second.h;
                    }

                    if (!node.is_leaf && feature_val < node.split_nbr.split_vals[k]) {
#pragma omp atomic
                        node.split_nbr.gain[k].lch_g += shift_it->second.g;
#pragma omp atomic
                        node.split_nbr.gain[k].lch_h += shift_it->second.h;
                    } else {
#pragma omp atomic
                        node.split_nbr.gain[k].rch_g += shift_it->second.g;
#pragma omp atomic
                        node.split_nbr.gain[k].rch_h += shift_it->second.h;
                    }
                }
            }
            node.calc_weight_(param.lambda, tree_ptr->g_bin_width, tree_ptr->h_bin_width);     // this lambda should be consistent with the training

            auto end_time_step_2_1 = clock::now();
            duration = end_time_step_2_1 - start_time_step_2_1;
            LOG(DEBUG) << "[Removing time] Level " << depth << " Size " << marginal_shifts_in_node.size() << " Step 2.1 (adjust gh according to marginal) = " << duration.count();

            if (node.is_leaf) continue;

            auto start_time_step_2_2 = clock::now();
            // recalculate the gain of each split neighbor
            for (int j = 0; j < node.split_nbr.split_bids.size(); ++j) {
                node.split_nbr.gain[j].gain_value = node.split_nbr.gain[j].cal_gain_value();
            }

            // update the best gain
            int old_best_idx = node.split_nbr.best_idx;
            node.split_nbr.update_best_idx_(invalid_bins);      // update without invalid bins
            node.gain = node.split_nbr.best_gain();
            float_type old_split_value = node.split_value;
            node.split_value = node.split_nbr.best_split_value();
            node.split_bid = node.split_nbr.best_bid();

            // recalculate default direction
            if (node.default_right) {
                node.gain.gain_value = -node.gain.gain_value;
                assert(node.gain.gain_value <= 0);
                DeltaTree::DeltaGain default_left_gain(node.gain);
                default_left_gain.lch_g += node.gain.missing_g;
                default_left_gain.lch_h += node.gain.missing_h;
                default_left_gain.rch_g -= node.gain.missing_g;
                default_left_gain.rch_h -= node.gain.missing_h;
                default_left_gain.gain_value = default_left_gain.cal_gain_value();
                if (ft_ge(fabs(default_left_gain.gain_value), fabs(node.gain.gain_value), 1e-2)) {
                    // switch default direction to left (marginal default left)
                    node.gain = default_left_gain;
                    node.default_right = false;
                }
            } else {
                node.gain.gain_value = node.gain.gain_value;
                assert(node.gain.gain_value >= 0);
                DeltaTree::DeltaGain default_right_gain(node.gain);
                default_right_gain.rch_g += node.gain.missing_g;
                default_right_gain.rch_h += node.gain.missing_h;
                default_right_gain.lch_g -= node.gain.missing_g;
                default_right_gain.lch_h -= node.gain.missing_h;
                default_right_gain.gain_value = -default_right_gain.cal_gain_value();
                if (!ft_ge(fabs(node.gain.gain_value), fabs(default_right_gain.gain_value), 1e-2)) {
                    // switch default direction to right (marginal default left)
                    default_right_gain.gain_value = -default_right_gain.gain_value;
                    node.gain = default_right_gain;
                    node.default_right = true;
                }
            }
            auto end_time_step_2_2 = clock::now();
            duration = end_time_step_2_2 - start_time_step_2_2;
            LOG(DEBUG) << "[Removing time] Level " << depth << " Step 2.2 (update best gain) = " << duration.count();

            auto start_time_step_2_3 = clock::now();
            // recalculate indices to be adjusted in the next layer
            int cur_best_idx = node.split_nbr.best_idx;
            std::unordered_map<int, GHPair> next_marginal_shift_left, next_marginal_shift_right;

            // record marginal instances
            // |---right node-----|----marginal instances---|-----left node------|
            // (the split indices is sorted in descending order of feature values)
            if (old_best_idx < cur_best_idx) {
                auto added_marginal_indices = flatten<int>(node.split_nbr.marginal_indices.begin() + old_best_idx,
                                             node.split_nbr.marginal_indices.begin() + cur_best_idx);
#pragma omp parallel for
                for (int j = 0; j < added_marginal_indices.size(); ++j) {
                    int iid = added_marginal_indices[j];
                    if (is_iid_removed[iid]) continue;
//                    next_marginal_shift_left[iid] =  -gh_pairs[iid];
//                    next_marginal_shift_right[iid] = gh_pairs[iid];
                    next_marginal_shift_left.insert({iid, -gh_pairs[iid]});
                    next_marginal_shift_right.insert({iid, gh_pairs[iid]});
                }
            } else if (old_best_idx > cur_best_idx) {
                auto added_marginal_indices = flatten<int>(node.split_nbr.marginal_indices.begin() + cur_best_idx,
                                                           node.split_nbr.marginal_indices.begin() + old_best_idx);
#pragma omp parallel for
                for (int j = 0; j < added_marginal_indices.size(); ++j) {
                    int iid = added_marginal_indices[j];
                    if (is_iid_removed[iid]) continue;
//                    next_marginal_shift_left[iid] = gh_pairs[iid];
//                    next_marginal_shift_right[iid] = -gh_pairs[iid];
                    next_marginal_shift_left.insert({iid, gh_pairs[iid]});
                    next_marginal_shift_right.insert({iid, -gh_pairs[iid]});
                }
            }
            auto end_time_step_2_3 = clock::now();
            duration = end_time_step_2_3 - start_time_step_2_3;
            LOG(DEBUG) << "[Removing time] Level " << depth << " Step 2.3 (calculate marginal indices for the next layer) = " << duration.count();

//            GHPair left_acc1 = std::accumulate(next_marginal_shift_left.begin(), next_marginal_shift_left.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });
//            GHPair right_acc1 = std::accumulate(next_marginal_shift_right.begin(), next_marginal_shift_right.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });
//            GHPair base_acc1 = std::accumulate(marginal_shifts_in_node.begin(), marginal_shifts_in_node.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });

            auto start_time_step_2_4 = clock::now();
            // merge these the marginal gh in this node into the next_marginal_gh_left and next_marginal_gh_right
#pragma omp parallel for
            for (int j = 0; j < marginal_shifts_in_node.size(); ++j) {
                auto shift_it = marginal_shifts_in_node.begin();
                std::advance(shift_it, j);  // move to the index j of map
                int iid = shift_it->first;
                bool is_missing = false;
//                float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);
                float_type feature_val = get_val_dense(iid, node.split_feature_id);
//                if (std::find(ins2node_indices[iid].begin(), ins2node_indices[iid].end(), node.lch_index) != ins2node_indices[iid].end()) {
                if (feature_val < node.split_nbr.best_split_value()) {
                    // this iid goes left
                    if (next_marginal_shift_left.count(iid)) {
#pragma omp atomic
                        next_marginal_shift_left[iid].g += shift_it->second.g;
#pragma omp atomic
                        next_marginal_shift_left[iid].h += shift_it->second.h;
                    } else {
//                        next_marginal_shift_left[iid] = shift_it->second;
                        next_marginal_shift_left.insert({iid, shift_it->second});
                    }
                } else {
                    // this iid goes right
                    if (next_marginal_shift_right.count(iid)) {
#pragma omp atomic
                        next_marginal_shift_right[iid].g += shift_it->second.g;
#pragma omp atomic
                        next_marginal_shift_right[iid].h += shift_it->second.h;
                    } else {
//                        next_marginal_shift_right[iid] = shift_it->second;
                        next_marginal_shift_right.insert({iid, shift_it->second});
                    }
                }
            }

            auto end_time_step_2_4 = clock::now();
            duration = end_time_step_2_4 - start_time_step_2_4;
            LOG(DEBUG) << "[Removing time] Level " << depth << " Step 2.4 (merge marginal indices) = " << duration.count();

//            GHPair left_acc2 = std::accumulate(next_marginal_shift_left.begin(), next_marginal_shift_left.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });
//            GHPair right_acc2 = std::accumulate(next_marginal_shift_right.begin(), next_marginal_shift_right.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });
//            GHPair base_acc2 = std::accumulate(marginal_shifts_in_node.begin(), marginal_shifts_in_node.end(), GHPair(), [](auto &a, auto &b){
//                return a + b.second;
//            });

            // add indices of left and right children
            assert(node.lch_index > 0 && node.rch_index > 0);
//            next_visiting_node_indices.push_back(node.lch_index);
//            next_visiting_node_indices.push_back(node.rch_index);
            next_visiting_node_indices[2 * i] = node.lch_index;
            next_visiting_node_indices[2 * i + 1] = node.rch_index;
            next_marginal_shifts[2 * i] = next_marginal_shift_left;
            next_marginal_shifts[2 * i + 1] = next_marginal_shift_right;

            auto end_time_in_node = clock::now();
            duration = end_time_in_node - start_time_in_node;
            LOG(DEBUG) << "[Removing time] Level " << depth << " Step 2 (in node) = " << duration.count();
        }
        clean_vectors_by_indices_(next_marginal_shifts, next_visiting_node_indices);
        clean_indices_(next_visiting_node_indices);     // clean lastly
        visit_node_indices = next_visiting_node_indices;
        marginal_shifts = next_marginal_shifts;
        assert(visit_node_indices.size() == marginal_shifts.size());
        auto end_time_level = clock::now();
        duration = end_time_level - start_time_level;
        LOG(DEBUG) << "Level " << depth << " finished, time = " << duration.count();
        depth++;
    }

    auto overall_end_time_step2 = clock::now();
    duration = overall_end_time_step2 - overall_start_time_step2;
    LOG(DEBUG) << "[Removing time] Step 2 (split point shifting) = " << duration.count();
}

void DeltaTreeRemover::get_invalid_sp(const vector<int> &dense_bin_id, const RobustHistCut& cut, const vector<int> &removed_indices,
                                      vector<vector<int>> &invalid_bids) {
    size_t n_features = dataSet->n_features();

    // initialize #removed instances in each bin as 0
    vector<vector<int>> n_remove_in_bins(n_features);
#pragma omp parallel for
    for (int fid = 0; fid < n_features; ++fid) {
        n_remove_in_bins[fid] = vector<int>(cut.n_instances_in_hist[fid].size(), 0);
    }

    auto n_remain_in_bins = cut.n_instances_in_hist;

    invalid_bids.resize(n_features);
#pragma omp parallel for
    for (int fid = 0; fid < n_features; ++fid) {
#pragma omp parallel for
        for (int i = 0; i < removed_indices.size(); ++i) {
            int iid = removed_indices[i];
            int bid = dense_bin_id[iid * n_features + fid];
#pragma omp atomic
            n_remove_in_bins[fid][bid]++;
#pragma omp atomic
            n_remain_in_bins[fid][bid]--;
        }
    }

    LOG(DEBUG);
}

void DeltaTreeRemover::get_invalid_sp(DeltaCut &cut, const DataSet &dataset, const vector<int>& removed_indices, int max_bin_size) {
    /**
     * Get invalid split points, stored in invalid_bins.
     * cut: the original DeltaCut to be removed from
     * removed_indices: the indices of instances to be removed
     * max_bin_size: the maximum bin size allowed
     * invalid_bins: fid x bid -> true if the bin is invalid else false
     */

    // extract removed values for each feature
    vector<vector<float_type>> removed_values(dataset.n_features(), vector<float_type>(removed_indices.size()));
#pragma omp parallel for
    for (int fid = 0; fid < dataset.n_features(); ++fid) {
        auto removed_value_begin = dataset.csc_val.begin() + dataset.csc_col_ptr[fid];
#pragma omp parallel for
        for (int i = 0; i < removed_indices.size(); ++i) {
            int iid = removed_indices[i];
            removed_values[fid][i] = *(removed_value_begin + iid);
        }
    }

    // remove values from cut
    vector<std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int>>>> invalid_bin_vec(dataset.n_features());
#pragma omp parallel for
    for (int fid = 0; fid < cut.n_features(); ++fid) {
        cut.bin_trees[fid].remove_instances_(removed_values[fid]);
        cut.bin_trees[fid].prune_(max_bin_size);
        cut.bin_trees[fid].trim_empty_bins_();
        vector<float_type> split_values_after_removal;
        cut.bin_trees[fid].get_split_values(split_values_after_removal);
        // cut.cut_points_val still remains unchanged
        vector<float_type> split_values_original(cut.cut_points_val.begin() + cut.cut_col_ptr[fid],
                                                 cut.cut_points_val.begin() + cut.cut_col_ptr[fid + 1]);

        // check if each split value in split_values_original is in split_values_after_removal,
        // store the result in invalid_bins
        int j = 0;
        for (int i = 0; i < split_values_original.size(); ++i) {
            if (!ft_eq(split_values_original[i], split_values_after_removal[j], 1e-6)) {
                invalid_bin_vec[fid].insert({fid, i}); //seg fault here
            } else {
                j++;
            }
        }
        if (j != split_values_after_removal.size()) {
            LOG(WARNING) << "Some split values in split_values_after_removal do not exist in split_values_original";
        }
    }

    // merge the invalid bins from different threads
    invalid_bins.clear();
    for (int fid = 0; fid < invalid_bin_vec.size(); ++fid) {
        invalid_bins.insert(invalid_bin_vec[fid].begin(), invalid_bin_vec[fid].end());
    }

}

void DeltaTreeRemover::prune() {
    /**
     * Prune invalid nodes, including
     * 1. If the weight of a node is less param.min_child_weight, set the node to invalid and its parent to leaf.
     *    (this scenario would not happen if all the bins are non-empty)
     * 2. If the gain of an internal node is less than param.rt_eps, set the node to leaf and its subtrees to invalid.
     */
    auto &nodes = tree_ptr->nodes;
    for (int nid = 0; nid < nodes.size(); ++nid) {
        auto &node = nodes[nid];
        if (!node.is_valid) continue;
//        if (node.gain.self_h < param.min_child_weight) {
//            node.is_valid = false;
//            assert(node.parent_index != -1);
//            auto parent = nodes[node.parent_index];
//            parent.is_leaf = true;
//        }
        if (!node.is_leaf && node.gain.gain_value < param.rt_eps) {
            node.is_leaf = true;

            // set all children to invalid  (optional)
            vector<int> visit = {node.lch_index, node.rch_index};
            while (!visit.empty()) {
                int cur_nid = visit.back();
                visit.pop_back();
                auto &cur_node = nodes[cur_nid];
                cur_node.is_valid = false;
                if (!cur_node.is_leaf) {
                    visit.push_back(cur_node.lch_index);
                    visit.push_back(cur_node.rch_index);
                }
            }
        }
    }
}





