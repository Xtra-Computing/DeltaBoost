//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/Tree/delta_tree_builder.h"
#include "thrust/sequence.h"
#include "thrust/iterator/discard_iterator.h"
#include "openssl/md5.h"
#include <algorithm>
#include <utility>
#include <numeric>
#include <random>
#include <unordered_set>

void DeltaTreeBuilder::init(DataSet &dataset, const DeltaBoostParam &param, bool skip_get_bin_ids) {
    TreeBuilder::init(dataset, param); // NOLINT(bugprone-parent-virtual-call)
    if (dataset.n_features() > 0) {
//        RobustHistCut ref_cut;
//        ref_cut.get_cut_points_by_instance(sorted_dataset, param.max_num_bin, n_instances);
//        cut.get_cut_points_by_instance(sorted_dataset, param.max_num_bin, n_instances);

//        RobustHistCut ref_cut;
//        ref_cut.get_cut_points_by_feature_range_balanced(sorted_dataset, param.max_bin_size, n_instances);

//        last_hist.resize((2 << param.depth) * cut.cut_points_val.size());

        cut.generate_bin_trees_(sorted_dataset, param.max_bin_size);
        cut.update_cut_points_(&sorted_dataset);
        if (!skip_get_bin_ids)
            get_bin_ids();
    }
    this->param = param;
    this->sp = SyncArray<DeltaSplitPoint>();
    update_random_feature_rank_(param.seed);
    update_random_split_nbr_rank_(param.seed);
}

void DeltaTreeBuilder::reset(DataSet &dataset, const DeltaBoostParam &param) {
    TreeBuilder::init(dataset, param); // NOLINT(bugprone-parent-virtual-call)
    get_bin_ids();
    this->param = param;
    this->sp = SyncArray<DeltaSplitPoint>();
    update_random_feature_rank_(param.seed);
    update_random_split_nbr_rank_(param.seed);
}


void DeltaTreeBuilder::init_nocutpoints(DataSet &dataset, const DeltaBoostParam &param) {
    TreeBuilder::init_nosortdataset(dataset, param);
    this->param = param;
}

void DeltaTreeBuilder::broadcast_potential_node_indices(int node_id) {
    std::queue<int> prior_nodes_to_broadcast;
    prior_nodes_to_broadcast.push(node_id);
    while (!prior_nodes_to_broadcast.empty()) {
        auto node = tree.nodes[prior_nodes_to_broadcast.front()];
        prior_nodes_to_broadcast.pop();

        if (!node.is_valid || node.is_leaf) {
            continue;
        }

        for (int i = 1; i < node.potential_nodes_indices.size(); ++i) {
            int potential_node_id = node.potential_nodes_indices[i];
//            tree.nodes[potential_node_id].potential_nodes_indices = node.potential_nodes_indices;
            assert(tree.nodes[potential_node_id].potential_nodes_indices == node.potential_nodes_indices);
        }

        prior_nodes_to_broadcast.push(node.lch_index);
        prior_nodes_to_broadcast.push(node.rch_index);
    }
}

vector<DeltaTree> DeltaTreeBuilder::build_delta_approximate(const SyncArray<GHPair> &gradients,
                                                            std::vector<std::vector<int>>& ins2node_indices_in_tree,
                                                            const vector<bool>& is_subset_indices_in_tree,
                                                            bool update_y_predict) {
    vector<DeltaTree> trees(param.tree_per_round);
    TIMED_FUNC(timerObj);

    // initiate timer
    typedef std::chrono::high_resolution_clock timer;
    auto start_time = timer::now();
    auto end_time = timer::now();
    std::chrono::duration<double> duration = end_time - start_time;

    for (int k = 0; k < param.tree_per_round; ++k) {
        DeltaTree &tree_k = trees[k];
        float_type gain_coef = 0.;

        tree.g_bin_width = g_bin_width;
        tree.h_bin_width = h_bin_width;

//        this->ins2node_id.resize(n_instances);
        this->gradients.resize(n_all_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_all_instances));
        this->tree.init_CPU(this->gradients, param, gain_coef);
        this->tree.nodes[0].n_instances = n_instances;
//        delta_gain_eps = param.delta_gain_eps * gain_coef;
//        LOG(INFO) << "delta_gain_eps = " << delta_gain_eps;
        num_nodes_per_level.clear();
        ins2node_indices.clear();

        if (param.hash_sampling_round > 1) {
            // initialize ins2node_indices
            for (int i = 0; i < n_all_instances; ++i) {
                if (is_subset_indices_in_tree[i]) {
                    ins2node_indices.push_back({0});
                } else {
                    ins2node_indices.push_back({-1});
                }
            }
        } else {
            // initialize ins2node_indices
            for (int i = 0; i < n_all_instances; ++i) {
                ins2node_indices.push_back({0});
            }
        }


        // root node 0 must be prior node
        is_prior.clear();
        is_prior.push_back(true);

        for (int level = 0; level < param.depth; ++level) {
            start_time = timer::now();
            find_split(level);
            end_time = timer::now();
            duration = end_time - start_time;
            LOG(DEBUG) << "find_split time = " << duration.count();


            start_time = timer::now();
            update_tree();
            end_time = timer::now();
            duration = end_time - start_time;
            LOG(DEBUG) << "update_tree time = " << duration.count();

            start_time = timer::now();
            update_ins2node_indices();
            end_time = timer::now();
            duration = end_time - start_time;
            LOG(DEBUG) << "update_ins2node_indices time = " << duration.count();

            LOG(TRACE) << "gathering ins2node id";
            //get final result of the reset instance id to node id
            if (!has_split) {
                LOG(INFO) << "no splittable nodes, stop";
                break;
            }

            LOG(DEBUG) << "Number of nodes: " << tree.nodes.size();
        }

//        broadcast_potential_node_indices(0);    // can remove
        //here
//        tree.prune_self(param.gamma);
//        LOG(INFO) << "y_predict: " << y_predict;
        start_time = timer::now();
        if(update_y_predict)
            predict_in_training(k);
        end_time = timer::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "predict_in_training time = " << duration.count();


        tree_k.nodes = tree.nodes;
        tree_k.dense_bin_id = dense_bin_id.to_vec();
        start_time = timer::now();
        tree_k.cut = cut;
        tree_k.g_bin_width = g_bin_width;
        tree_k.h_bin_width = h_bin_width;
        end_time = timer::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "copy tree time = " << duration.count();

    }

    ins2node_indices_in_tree = ins2node_indices;    // return this value for removal
    return trees;
}

void DeltaTreeBuilder::find_split(int level) {

    // initialize timer
    typedef std::chrono::high_resolution_clock timer;
    auto start_time = timer::now();
    auto end_time = timer::now();
    std::chrono::duration<double> duration = end_time - start_time;

    int n_nodes_in_level = level == 0 ? 1 : num_nodes_per_level[level - 1] * 2;
//    int n_nodes_in_level = 1 << level;
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
    int n_bins = static_cast<int>(cut.cut_points_val.size());
    int n_max_nodes = n_nodes_in_level * 2;
    int n_max_splits = n_max_nodes * n_bins;
    vector<int> node_indices(n_nodes_in_level);
    std::iota(node_indices.begin(), node_indices.end(), tree.nodes.size() - n_nodes_in_level);

    auto cut_fid_data = cut.cut_fid.data();

//    auto i2fid = [=] __host__(int i) { return cut_fid_data[i % n_bins]; };
//    auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);

    int nid_start_idx = 0;
    for (int i = 0; i < level; ++i) {
        nid_start_idx += num_nodes_per_level[i];
    }

    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
    auto hist_fid_data = hist_fid.host_data();

#pragma omp parallel for
    for (int i = 0; i < hist_fid.size(); i++)
        hist_fid_data[i] = cut_fid_data[i % n_bins];


    int n_split = n_nodes_in_level * n_bins;
    SyncArray<GHPair> missing_gh(n_partition);
    vector<float_type> missing_g2(n_partition);
    LOG(TRACE) << "start finding split";

    SyncArray<GHPair> hist(n_max_splits);
    vector<float_type> hist_g2(n_max_splits);

    vector<DeltaTree::DeltaGain> gain(n_nodes_in_level * n_bins);

    start_time = timer::now();
    compute_histogram_in_a_level(level, n_max_splits, n_bins, n_nodes_in_level, hist_fid_data, missing_gh, hist, missing_g2, hist_g2);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "compute_histogram_in_a_level time = " << duration.count();

    start_time = timer::now();
    compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data, missing_gh, hist_g2, missing_g2, hist, 0);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "compute_gain_in_a_level time = " << duration.count();

    start_time = timer::now();
    vector<DeltaTree::SplitNeighborhood> best_split_nbr(n_nodes_in_level);
    get_best_split_nbr(gain, best_split_nbr, n_nodes_in_level, n_bins, param.nbr_size);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "get_best_split_nbr time = " << duration.count();

    start_time = timer::now();
    update_indices_in_split_nbr(best_split_nbr, node_indices);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "update_indices_in_split_nbr time = " << duration.count();

    start_time = timer::now();
    get_split_points(best_split_nbr, n_nodes_in_level, hist_fid_data, missing_gh, hist, hist_g2, missing_g2, level);
    end_time = timer::now();
    duration = end_time - start_time;
    LOG(DEBUG) << "get_split_points time = " << duration.count();

    num_nodes_per_level.emplace_back(n_nodes_in_level);
}

void DeltaTreeBuilder::get_topk_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                              int n_nodes_in_level, int n_bins, int k) {
    /**
     * gain: size of gain is equal to the number of bins (n_nodes_in_a_level x max_bins)
     * topk_idx_gain: the int refers to the index in the gain vector (i.e., bid)
     */
    auto arg_abs_max = [](const gain_pair &a, const gain_pair &b) {
        if (fabsf(a.second.gain_value) == fabsf(b.second.gain_value))
            return a.first < b.first;
        else
            return fabsf(a.second.gain_value) > fabsf(b.second.gain_value);
    };

    // make tuple with indices and gains
    vector<gain_pair> idx_gain;
    for (int i = 0; i < gain.size(); ++i) {
        idx_gain.emplace_back(std::make_pair(i, gain[i]));
    }

    for (int i = 0; i < n_bins * n_nodes_in_level; i += n_bins) {
        std::partial_sort(idx_gain.begin() + i, idx_gain.begin() + i + k,
                          idx_gain.begin() + i + n_bins, arg_abs_max);
        std::vector<gain_pair> topk_idx_gain_per_bin(idx_gain.begin() + i, idx_gain.begin() + i + k);
        topk_idx_gain.emplace_back(topk_idx_gain_per_bin);
    }
}

int DeltaTreeBuilder::get_threshold_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain,
                                                     vector<vector<gain_pair>> &potential_idx_gain, int n_nodes_in_level,
                                                     int n_bins, float_type min_diff, float_type max_range,
                                                     const vector<int> &n_samples_in_nodes) {
    /**
     * @param min_diff: the min difference between two gains. |gain1 - gain2| >= min_diff
     * @param max_range: the max tolerable range of gains. All gains should be in range [max_gain - max_range, max_gain]
     * @param n_samples_in_nodes: number of instances in nodes in this level (of size n_nodes_in_level)
     */
    auto arg_abs_max = [](const gain_pair &a, const gain_pair &b) {
        if (fabsf(a.second.gain_value) == fabsf(b.second.gain_value))
            return a.first < b.first;
        else
            return fabsf(a.second.gain_value) > fabsf(b.second.gain_value);
    };

    // make tuple with indices and gains
    vector<gain_pair> idx_gain(gain.size());
#pragma omp parallel for
    for (int i = 0; i < gain.size(); ++i) {
        idx_gain[i] = std::make_pair(i, gain[i]);
    }

    int total_size = 0;
    for (int i = 0; i < n_bins * n_nodes_in_level; i += n_bins) {
        int nid = i / n_bins;

        float_type min_diff_node = min_diff * (float)(n_samples_in_nodes[nid]) / (float)n_instances;
        float_type max_range_node = max_range * (float)(n_samples_in_nodes[nid]) / (float)n_instances;
//        assert(n_samples_in_nodes[nid] > 0);
        if (n_samples_in_nodes[nid] == 0)       // to be fixed. filter all empty nodes
            min_diff_node = 1.;

        vector<gain_pair> potential_idx_gain_per_node;
        std::sort(idx_gain.begin() + i, idx_gain.begin() + i + n_bins, arg_abs_max);
        float_type last_gain = std::abs(idx_gain[i].second.gain_value);
        potential_idx_gain_per_node.emplace_back(idx_gain[i]);
        for (int j = i + 1; j < i + n_bins; ++j) {
            if (fabs(idx_gain[j].second.gain_value) < fabs(idx_gain[i].second.gain_value) - max_range_node) {
                break;  // gain too small, unacceptable
            }

            if (fabs(idx_gain[j].second.gain_value) <= last_gain - min_diff_node) {
                last_gain = fabs(idx_gain[j].second.gain_value);
                potential_idx_gain_per_node.emplace_back(idx_gain[j]);
            }
        }

        potential_idx_gain.emplace_back(potential_idx_gain_per_node);
        total_size += static_cast<int>(potential_idx_gain_per_node.size());
    }

    return total_size;
}


void DeltaTreeBuilder::get_best_split_nbr(const vector<DeltaTree::DeltaGain> &gain,
                                          vector<DeltaTree::SplitNeighborhood> &best_split_nbr,
                                          int n_nodes_in_level, int n_bins, int nbr_size) {
    auto arg_abs_max = [](const gain_pair &a, const gain_pair &b) {
        if (std::abs(a.second.gain_value) == std::abs(b.second.gain_value))
            return a.first < b.first;
        else
            return std::abs(a.second.gain_value) > std::abs(b.second.gain_value);
    };

    // initialize timer
    typedef std::chrono::high_resolution_clock timer;
    auto start_time = timer::now();
    auto end_time = timer::now();
    std::chrono::duration<double> duration = end_time - start_time;
    // calculate score
    vector<float_type> gain_per_sp(gain.size());
//    vector<float_type> remain_gain_per_sp(gain.size());
#pragma omp parallel for
    for (int i = 0; i < gain.size(); ++i) {
        gain_per_sp[i] = std::abs(gain[i].gain_value);
//        remain_gain_per_sp[i] = gain[i].ev_remain_gain;
    }

    struct IdxScore {
        int idx_in_feature_start = 0;
        int idx_in_feature_end = 0;
        int fid = 0;
        float_type score = 0.l;
//        float_type remain_score = 0.l;

        bool operator<(const IdxScore &other) const {
            return score < other.score;
        }

        bool operator>(const IdxScore &other) const {
            return score > other.score;
        }
    };

#pragma omp parallel for
    for (int i = 0; i < n_bins * n_nodes_in_level; i += n_bins) {
        int nid = i / n_bins;

        // choose the best split neighborhood (with max scores)
        vector<IdxScore> idx_scores(sorted_dataset.n_features());
        start_time = timer::now();
#pragma omp parallel for
        for (int j = 0; j < sorted_dataset.n_features(); ++j) {
            int bid_start = cut.cut_col_ptr[j];
            int bid_end = cut.cut_col_ptr[j + 1];
            int n_nbrs = bid_end - bid_start - nbr_size + 1;
            vector<IdxScore> idx_scores_in_feature;
//            vector<float_type> scores_in_feature;
//            vector<float_type> remain_scores_in_feature;

            if (n_nbrs > 0){
                idx_scores_in_feature.resize(n_nbrs);
//                scores_in_feature.resize(n_nbrs);
//                remain_scores_in_feature.resize(n_nbrs);
                for (int k = bid_start; k < bid_end - nbr_size + 1; ++k) {
                    float_type score = std::accumulate(gain_per_sp.begin() + i + k, gain_per_sp.begin() + i + k + nbr_size, 0.);
//                    float_type remain_score = std::accumulate(remain_gain_per_sp.begin() + i + k, remain_gain_per_sp.begin() + i + k + nbr_size, 0.);
                    idx_scores_in_feature[k - bid_start].idx_in_feature_start = k;
                    idx_scores_in_feature[k - bid_start].idx_in_feature_end = std::min(k + nbr_size, bid_end);
                    idx_scores_in_feature[k - bid_start].fid = j;
                    idx_scores_in_feature[k - bid_start].score = score / static_cast<float_type>(nbr_size);
//                    idx_scores_in_feature[k - bid_start].remain_score = remain_score / static_cast<float_type>(nbr_size);
                }
            } else {
                idx_scores_in_feature.resize(1);
                float_type score = std::accumulate(gain_per_sp.begin() + i + bid_start, gain_per_sp.begin() + i + bid_end, 0.);
//                float_type remain_score = std::accumulate(remain_gain_per_sp.begin() + i + bid_start, remain_gain_per_sp.begin() + i + bid_end, 0.);
                if (bid_start != bid_end) {
                    idx_scores_in_feature[0].score = score / static_cast<float_type>(bid_end - bid_start);
//                    idx_scores_in_feature[0].remain_score = remain_score / static_cast<float_type>(bid_end - bid_start);
                } else {
//                    idx_scores_in_feature[0].score = idx_scores_in_feature[0].remain_score = 0.;
                }
                idx_scores_in_feature[0].idx_in_feature_start = bid_start;
                idx_scores_in_feature[0].idx_in_feature_end = bid_end;
                idx_scores_in_feature[0].fid = j;
            }

            int max_candidate_features = 100;

            if (idx_scores_in_feature.size() < max_candidate_features) {
                std::sort(idx_scores_in_feature.begin(), idx_scores_in_feature.end(), std::greater<>());
                max_candidate_features = idx_scores_in_feature.size();
            } else {
                std::partial_sort(idx_scores_in_feature.begin(), idx_scores_in_feature.begin() + max_candidate_features, idx_scores_in_feature.end(), std::greater<>());
            }

            // find the first gap larger than delta_gain_eps
            int gap_idx;
            for (gap_idx = 0; gap_idx < max_candidate_features - 1; ++gap_idx) {
                if (idx_scores_in_feature[gap_idx].score - idx_scores_in_feature[gap_idx+1].score > param.delta_gain_eps_sn)
                    break;
            }

            // choose the first split_nbr according to random feature order
            auto best_idx_score_itr = std::max_element(idx_scores_in_feature.begin(),
                                                       idx_scores_in_feature.begin() + gap_idx + 1, [&](const auto &a, const auto &b){
                return RobustHistCut::cut_value_hash_comp(cut.cut_points_val[a.idx_in_feature_start], cut.cut_points_val[b.idx_in_feature_start]);
            });
            idx_scores[j] = *best_idx_score_itr;

        }

        end_time = timer::now();
        duration = end_time - start_time;
//        if (duration.count() > 0.001)
            LOG(DEBUG) << "Node " << nid << " get_best_split_nbr inner time: " << duration.count();

        // assert scores >= 0
//        auto best_idx_score_itr = std::max_element(idx_scores.begin(), idx_scores.end(), [](const auto &a, const auto &b){
//            return std::get<2>(a) < std::get<2>(b);
//        });
//        int fid = static_cast<int>(best_idx_score_itr - idx_scores.begin());
//        int best_bid_start = static_cast<int>(std::get<0>(*best_idx_score_itr));
//        int best_bid_end = static_cast<int>(std::get<1>(*best_idx_score_itr));
//        float_type best_score = std::get<2>(*best_idx_score_itr);
//        float_type remain_best_score = std::get<3>(*best_idx_score_itr);

//        // get the second feature, check if the best feature is robust
//        bool is_robust = true;

    //        for (int j = 0; j < idx_scores.size(); ++j) {
//            if (j != fid && best_score - std::get<2>(idx_scores[j]) < param.delta_gain_eps &&
//                         remain_best_score - std::get<3>(idx_scores[j]) < param.delta_gain_eps) {
//                is_robust = false;
//            }
//        }
//
//        if (remain_best_score < param.delta_gain_eps || best_score < param.delta_gain_eps) {
//            is_robust = false;
//        }

        std::sort(idx_scores.begin(), idx_scores.end(), std::greater<>());

        // find the first gap larger than delta_gain_eps
        int gap_idx;
        for (gap_idx = 0; gap_idx < idx_scores.size() - 1; ++gap_idx) {
            if (idx_scores[gap_idx].score - idx_scores[gap_idx+1].score > param.delta_gain_eps_feature)
                break;
        }

        // choose the first split_nbr according to random feature order
        auto best_idx_score_itr = std::max_element(idx_scores.begin(), idx_scores.begin() + gap_idx + 1, [&](const auto &a, const auto &b){
            return random_feature_rank[a.fid] < random_feature_rank[b.fid];
        });

        bool is_robust = true;
        if (best_idx_score_itr->score < param.min_gain) {
            is_robust = false;
        }

        if (is_robust) {
            // extract best split neighborhood according to best_bid
            vector<int> best_bid_vec(best_idx_score_itr->idx_in_feature_end - best_idx_score_itr->idx_in_feature_start);
            std::iota(best_bid_vec.begin(), best_bid_vec.end(), best_idx_score_itr->idx_in_feature_start);
            vector<DeltaTree::DeltaGain> best_gain_vec(gain.begin() + i + best_idx_score_itr->idx_in_feature_start, gain.begin() + i + best_idx_score_itr->idx_in_feature_end);
            vector<float_type> best_split_val_vec(
                    cut.get_cut_point_val_itr(best_bid_vec[0]),       // global bid
                    cut.get_cut_point_val_itr(best_bid_vec[best_bid_vec.size() - 1] + 1));      // global bid
            DeltaTree::SplitNeighborhood split_nbr(best_bid_vec, best_idx_score_itr->fid, best_gain_vec, best_split_val_vec);
            split_nbr.update_best_idx_();
            best_split_nbr[nid] = split_nbr;
        } else {
            // generate a split_nbr with gain 0, forcing tree to stop splitting
            best_split_nbr[nid] = DeltaTree::SplitNeighborhood();
        }

    }
}



int DeltaTreeBuilder::filter_potential_idx_gain(const vector<vector<gain_pair>>& candidate_idx_gain,
                               vector<vector<gain_pair>>& potential_idx_gain,
                               float_type quantized_width, int max_num_potential) {
    int total_size = 0;
    for (const auto& idx_gain_list: candidate_idx_gain) {
        vector<gain_pair> potential_idx_gain_per_node;
        if (idx_gain_list.empty()) {
            potential_idx_gain.emplace_back(potential_idx_gain_per_node);
            continue;
        }

        potential_idx_gain_per_node.emplace_back(idx_gain_list[0]);
        double last_gain = fabs(idx_gain_list[0].second.gain_value);
        for (size_t i = 1; i < idx_gain_list.size(); ++i) {
            if (fabs(idx_gain_list[i].second.gain_value) > last_gain - quantized_width) {
                continue;
            } else {
                last_gain = fabs(idx_gain_list[i].second.gain_value);
                potential_idx_gain_per_node.emplace_back(idx_gain_list[i]);
            }

            if (potential_idx_gain_per_node.size() >= max_num_potential) {
                break;
            }
        }
        potential_idx_gain.emplace_back(potential_idx_gain_per_node);
        total_size += potential_idx_gain_per_node.size();
    }
    return total_size;
}


//todo: reduce hist size according to current level (not n_max_split)
void DeltaTreeBuilder::compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                                   int *hist_fid, SyncArray<GHPair> &missing_gh,
                                                   SyncArray<GHPair> &hist, vector<float_type> &missing_g2,
                                                   vector<float_type> &hist_g2) {
    std::chrono::high_resolution_clock timer;

//    const SyncArray<int> &nid = ins2node_id;
    const auto &gh_pair = gradients;
//    DeltaTree &tree = this->tree;
    const auto &sp = this->sp;
    const auto &cut = this->cut;
    const auto &dense_bin_id = this->dense_bin_id;
    int nid_offset = tree.nodes.size() - n_nodes_in_level;
//    auto &last_hist = this->last_hist;

    TIMED_FUNC(timerObj);
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    {   // redefine timerObj
        TIMED_SCOPE(timerObj, "build hist");
        int n_bids = 0;
        if (n_nodes_in_level == 1) {
            auto hist_data = hist.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr;
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
//            for (auto &gh: gh_pair.to_vec()) {
//                assert(gh.h >= 0);
//            }

            GHPair sum_gh0 = std::accumulate(hist_data, hist_data + cut.cut_col_ptr[1], GHPair());

//            vector<int> indices;
//            GHPair sum_gh5;
            vector<GHPair> gh_vec(cut.cut_col_ptr[1], GHPair());
#pragma omp parallel for
            for (int i = 0; i < n_all_instances * n_column; i++) {
                int iid = i / n_column;
                int fid = i % n_column;
                auto bid = dense_bin_id_data[iid * n_column + fid];
                if (bid != -1) {
                    int feature_offset = cut_col_ptr_data[fid];
//                    if (0 <= feature_offset + bid && feature_offset + bid < cut.cut_col_ptr[1]) {
//                        indices.push_back(iid);
//                        assert(feature_offset == 0);
//                        assert(fid == 0);
//                        gh_vec.at(bid) += gh_data[iid];
//                        sum_gh5 += gh_data[iid];
//                    } else {
//                        assert(fid != 0);
//                    }
#pragma omp atomic
                    hist_data[feature_offset + bid].g += gh_data[iid].g;
#pragma omp atomic
                    hist_data[feature_offset + bid].h += gh_data[iid].h;
//                    hist_g2[feature_offset + bid] += gh_data[iid].g * gh_data[iid].g;
                }
            }
//            GHPair sum_gh4 = std::accumulate(gh_vec.begin(), gh_vec.end(), GHPair());
            GHPair sum_gh1 = std::accumulate(hist_data, hist_data + cut.cut_col_ptr[1], GHPair());
            LOG(DEBUG);
        } else {
            auto t_dp_begin = timer.now();
            // sort the indices of instance s.t. each node can be split by node_ptr
            // E.g., node_idx=[1,3,5,7, 2,4,6,8], node_ptr=[4,8], represents two nodes. In node 1, the instances are
            // 1,3,5,7; in node 2, the instances are 2,4,6,8
//            SyncArray<int> node_idx(n_instances);
//            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            vector<vector<int>> nid_to_iid(tree.nodes.size(), vector<int>());
            {
                TIMED_SCOPE(timerObj, "data partitioning");
                for (int i = 0; i < n_all_instances; ++i) {
                    for (int node_id: ins2node_indices[i]) {
                        if (node_id != -1)
                            nid_to_iid[node_id].emplace_back(i);
                    }
                }
            }
            auto t_dp_end = timer.now();
            std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
            this->total_dp_time += dp_used_time.count();


//            auto node_ptr_data = node_ptr.host_data();
//            auto node_idx_data = node_idx.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
//            auto max_num_bin = param.max_num_bin;


            for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                int nid0_to_compute = i * 2;
                int nid0_to_substract = i * 2 + 1;
                //node_ptr_data[i+1] - node_ptr_data[i] is the number of instances in node i, i is the node id in current level (start from 0)
//                int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
//                int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                int n_ins_left = nid_to_iid[nid0_to_compute + nid_offset].size();
                int n_ins_right = nid_to_iid[nid0_to_substract + nid_offset].size();
                if (std::max(n_ins_left, n_ins_right) == 0) continue;
                //only compute the histogram on the node with the smaller data
                if (n_ins_left > n_ins_right)
                    std::swap(nid0_to_compute, nid0_to_substract);
                 //compute histogram
                {
                    int nid0 = nid0_to_compute + nid_offset;
//                    auto idx_begin = node_ptr.host_data()[nid0];
//                    auto idx_end = node_ptr.host_data()[nid0 + 1];
                    auto hist_data = hist.host_data() + nid0_to_compute * n_bins;
                    this->total_hist_num++;

#pragma omp parallel for
                    for (int j = 0; j < nid_to_iid[nid0].size() * n_column; j++) {
                        int iid = nid_to_iid[nid0][j / n_column];
                        int fid = j % n_column;
                        int bid = dense_bin_id_data[iid * n_column + fid];
                        assert(iid >= 0);

                        if (bid != -1) {
                            int feature_offset = cut_col_ptr_data[fid];
#pragma omp atomic
                            hist_data[feature_offset + bid].g += gh_data[iid].g;
#pragma omp atomic
                            hist_data[feature_offset + bid].h += gh_data[iid].h;
//                            hist_g2[nid0_to_compute * n_bins + feature_offset + bid] += gh_data[iid].g * gh_data[iid].g;
                        }
                    }
//                    LOG(DEBUG);
                }

                //subtract to the histogram of the other node
                auto t_copy_start = timer.now();
                {
                    auto hist_data_computed = hist.host_data() + nid0_to_compute * n_bins;
                    auto hist_data_to_compute = hist.host_data() + nid0_to_substract * n_bins;
                    auto parent_hist_data = last_hist.host_data() + parent_indices[nid0_to_substract] * n_bins;
                    auto hist_g2_computed = hist_g2.data() + nid0_to_compute * n_bins;
                    auto hist_g2_to_compute = hist_g2.data() + nid0_to_substract * n_bins;
//                    auto parent_hist_g2 = last_hist_g2.data() + parent_indices[nid0_to_substract] * n_bins;
#pragma omp parallel for
                    for (int j = 0; j < n_bins; j++) {
                        hist_data_to_compute[j] = parent_hist_data[j] - hist_data_computed[j];
//                        hist_g2_to_compute[j] = parent_hist_g2[j] - hist_g2_computed[j];
                    }
                }
                auto t_copy_end = timer.now();
                std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
            }  // end for each node
        }
        last_hist.resize(n_nodes_in_level * n_bins);
//        last_hist_g2.resize(n_nodes_in_level * n_bins);
        auto last_hist_data = last_hist.host_data();
        const auto hist_data = hist.host_data();
#pragma omp parallel for
        for (int i = 0; i < n_nodes_in_level * n_bins; i++) {
            last_hist_data[i] = hist_data[i];
//            last_hist_g2[i] = hist_g2[i];
        }
    }

//    float_type sum_g21 = std::accumulate(gh_pair.host_data(), gh_pair.host_data() + gh_pair.size(), 0.0, [](float_type a, const GHPair &b){ return  a + b.g * b.g;});
//    float_type sum_g22 = std::accumulate(hist_g2.begin(), hist_g2.begin() + cut.cut_col_ptr[1], 0.0);

    this->build_n_hist++;
//    inclusive_scan_by_key(thrust::host, hist_fid, hist_fid + n_split,
//                          hist.host_data(), hist.host_data());
    inclusive_scan_by_cut_points(hist.host_data(), cut.cut_col_ptr.data(),
                                 n_nodes_in_level, n_column, n_bins, hist.host_data());

//    sum_g21 = std::accumulate(gh_pair.host_data(), gh_pair.host_data() + gh_pair.size(), 0.0, [](float_type a, const GHPair &b){ return  a + b.g * b.g;});
//    sum_g22 = hist_g2[cut.cut_col_ptr[1] - 1];
//    LOG(DEBUG) << hist;

    // handle missing data
    auto &nodes_data = tree.nodes;
    auto missing_gh_data = missing_gh.host_data();

    auto cut_col_ptr = cut.cut_col_ptr.data();
    auto hist_data = hist.host_data();

#pragma omp parallel for
    for (int pid = 0; pid < n_partition; pid++) {
//        int nid0 = pid / n_column;
//        int nid = nid0 + nid_offset;
//        if (!nodes_data[nid].splittable()) continue;
//        int fid = pid % n_column;
//        if (cut_col_ptr[fid + 1] != cut_col_ptr[fid]) {
//            GHPair node_gh = hist_data[nid0 * n_bins + cut_col_ptr[fid + 1] - 1];
//            float_type node_g2 = hist_g2[nid0 * n_bins + cut_col_ptr[fid + 1] - 1];
//            missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
////            if (missing_gh_data[pid].h < 0) {
////                vector<float_type> close_h;
////
////            }
////            assert(missing_gh_data[pid].h >= 0);
//            missing_g2[pid] = nodes_data[nid].sum_g2 - node_g2;
//            // missing value should be zero.
//        }
        missing_gh_data[pid] = GHPair(0, 0);
    }
}

void DeltaTreeBuilder::update_ins2node_indices() {
    // update indices_in_node simultaneously
    TIMED_FUNC(timerObj);

//    auto nid_data = ins2node_id.host_data();
    SyncArray<bool> has_splittable(1);
//    auto &columns = shards.columns;
    //set new node id for each instance
    {
//        TIMED_SCOPE(timerObj, "get new node id");
        auto &nodes_data = tree.nodes;
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.host_data();

        int n_column = sorted_dataset.n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
//        int max_num_bin = param.max_num_bin;
#pragma omp parallel for
        for (int iid = 0; iid < n_all_instances; iid++) {
            for (int j = 0; j < ins2node_indices[iid].size(); ++j) {
                int nid = ins2node_indices[iid][j];
                if (nid == -1)  // this instance does not exist in this node
                    continue;
                const auto &node = nodes_data[nid];
                int split_fid = node.split_feature_id;
                if (node.splittable() && ((split_fid < n_column) && (split_fid >= 0))) {
                    h_s_data[0] = true;
                    auto split_bid = node.split_bid;
                    auto bid = dense_bin_id_data[iid * n_column + split_fid];
                    bool to_left;
                    if (bid == -1) {
                        to_left = !node.default_right;
                    } else {
                        to_left = bid > split_bid;
                    }
                    if (to_left) {
                        //goes to left child
                        ins2node_indices[iid][j] = node.lch_index;
//                        if (is_prior[nid]) {
//                            nid_data[iid] = node.lch_index;
//                        }

                    #pragma omp atomic
                        nodes_data[node.lch_index].n_instances += 1;
                    } else {
                        //right child
                        ins2node_indices[iid][j] = node.rch_index;
//                        if (is_prior[nid]) {
//                            nid_data[iid] = node.rch_index;
//                        }
                    #pragma omp atomic
                        nodes_data[node.rch_index].n_instances += 1;
                    }
                }
            }
        }
    }
    has_split = has_splittable.host_data()[0];
//    LOG(DEBUG) << "new tree_id = " << ins2node_id;

}



[[deprecated]]
void DeltaTreeBuilder::update_ins2node_id() {
    // update indices_in_node simultaneously
    TIMED_FUNC(timerObj);

    SyncArray<bool> has_splittable(1);
//    auto &columns = shards.columns;
    //set new node id for each instance
    {
//        TIMED_SCOPE(timerObj, "get new node id");
        auto nid_data = ins2node_id.host_data();
        auto &nodes_data = tree.nodes;
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.host_data();

        int n_column = sorted_dataset.n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
//        int max_num_bin = param.max_num_bin;
#pragma omp parallel for
        for (int iid = 0; iid < n_instances; iid++) {
            int nid = nid_data[iid];
            const DeltaTree::DeltaNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid < n_column) && (split_fid >= 0))) {
                h_s_data[0] = true;
                auto split_bid = node.split_bid;
                auto bid = dense_bin_id_data[iid * n_column + split_fid];
                bool to_left;
                if (bid == -1) {
                    to_left = !node.default_right;
                } else {
                    to_left = bid > split_bid;
                }
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                    #pragma omp atomic
                    nodes_data[node.lch_index].n_instances += 1;

                    for (int &potential_nid: ins2node_indices[iid]) {
                        if (potential_nid >= 0 && !nodes_data[potential_nid].is_leaf) {
                            potential_nid = nodes_data[potential_nid].lch_index;
                        }
                    }
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                    #pragma omp atomic
                    nodes_data[node.rch_index].n_instances += 1;

                    for (int &potential_nid: ins2node_indices[iid]) {
                        if (potential_nid >= 0 && !nodes_data[potential_nid].is_leaf) {
                            potential_nid = nodes_data[potential_nid].rch_index;
                        }
                    }
                }
            }
        }
    }
    LOG(DEBUG) << "new tree_id = " << ins2node_id;
    has_split = has_splittable.host_data()[0];
}


void DeltaTreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
//    auto& sp = this->sp;
//    auto& tree = this->tree;
    auto sp_data = sp.host_data();
    int n_nodes_in_level = sp.size();

//    auto &nodes_data = tree.nodes;
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

#pragma omp parallel for
    for(int i = 0; i < n_nodes_in_level; i++){
        DeltaTree::DeltaGain best_split_gain = sp_data[i].gain;

        if (fabs(best_split_gain.gain_value) > rt_eps) {
            //do split
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            DeltaTree::DeltaNode &node = tree.nodes[nid];
            node.gain = best_split_gain;

            DeltaTree::DeltaNode &lch = tree.nodes[node.lch_index];//left child
            DeltaTree::DeltaNode &rch = tree.nodes[node.rch_index];//right child
            lch.is_valid = node.is_valid;
            rch.is_valid = node.is_valid;
            node.split_feature_id = sp_data[i].split_fea_id;
            node.split_nbr = sp_data[i].split_nbr;
            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            float_type p_missing_g2 = sp_data[i].fea_missing_g2;
            //todo process begin
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
//            rch.sum_g2 = sp_data[i].rch_sum_g2;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
//                rch.sum_g2 = rch.sum_g2 + p_missing_g2;
                // LOG(INFO) << "RCH" << rch.sum_gh_pair;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
//            lch.sum_g2 = node.sum_g2 - rch.sum_g2;
            //  LOG(INFO) << "LCH" << lch.sum_gh_pair;
            lch.calc_weight_(lambda, g_bin_width, h_bin_width);
            rch.calc_weight_(lambda, g_bin_width, h_bin_width);
        } else {
            //set leaf
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            DeltaTree::DeltaNode &node = tree.nodes[nid];
            node.is_leaf = true;
            tree.nodes[node.lch_index].is_valid = false;
            tree.nodes[node.rch_index].is_valid = false;
            node.lch_index = -2;
            node.rch_index = -2;
        }
    }
    // LOG(INFO) << tree.nodes;
}


void DeltaTreeBuilder::predict_in_training(int k) {
    auto y_predict_data = y_predict.host_data() + k * n_instances;
//    auto nid_data = ins2node_id.host_data();
    vector<float_type> y_predict_vec = vector<float_type>(n_all_instances, 0);
    const auto &nodes_data = tree.nodes;
    auto lr = param.learning_rate;
#pragma omp parallel for
    for(int i = 0; i < n_all_instances; i++){
        int nid = ins2node_indices[i][0];
        if (nid == -1)
            continue;
        while (nid != -1 && nodes_data[nid].is_pruned) nid = nodes_data[nid].parent_index;
        y_predict_vec[i] += lr * nodes_data[nid].base_weight;
    }

    int j = 0;
    for (int i = 0; i < n_all_instances; i++) {
        if (ins2node_indices[i][0] != -1) {
            y_predict_data[j++] += y_predict_vec[i];
        }
    }
    assert(j == n_instances);

////    LOG(DEBUG) << y_predict;
//    auto y_predict_data = y_predict.host_data() + k * n_instances;
////    auto nid_data = ins2node_id.host_data();
//
//    const auto *nodes_data = tree.nodes.data();
//    auto lr = param.learning_rate;
////#pragma omp parallel for
//    for(int i = 0; i < n_instances; i++){
////        int nid = nid_data[i];
//        int nid = ins2node_indices[i][0];
//        assert(nid != -1);
//        while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
//        y_predict_data[i] += lr * nodes_data[nid].base_weight;
//    }
//    LOG(DEBUG) << y_predict;
}


void
DeltaTreeBuilder::compute_gain_in_a_level(vector<DeltaTree::DeltaGain> &gain, int n_nodes_in_level, int n_bins,
                                          int *hist_fid, SyncArray<GHPair> &missing_gh, vector<float_type> &hist_g2,
                                          vector<float_type> &missing_g2, SyncArray<GHPair> &hist, int n_column) {

    if (n_column == 0)
        n_column = sorted_dataset.n_features();
    int n_split = n_nodes_in_level * n_bins;
    const int nid_offset = num_nodes_per_level.empty() ? 0 :
                           std::accumulate(num_nodes_per_level.begin(), num_nodes_per_level.end(), 0);
//    auto compute_gain = []__host__(GHPair parent, GHPair lch, GHPair rch, float_type min_child_weight,
//                                   float_type lambda) -> float_type {
//        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
//            return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
//                   (parent.g * parent.g) / (parent.h + lambda);
//        else
//            return 0;
//    };
    const auto &nodes_data = tree.nodes;
    const GHPair *gh_prefix_sum_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
//    auto ignored_set_data = ignored_set.host_data();
    //for lambda expression
    float_type mcw = param.min_child_weight;
    float_type l = param.lambda;

#pragma omp parallel for  // comment for debug
    for (int i = 0; i < n_split; i++) {
        int nid0 = i / n_bins;
        int nid = nid0 + nid_offset;
        int fid = hist_fid[i % n_bins];
        if (nodes_data[nid].is_valid) {
            int pid = nid0 * n_column + fid;
            GHPair parent_gh = nodes_data[nid].sum_gh_pair;
//            float_type parent_g2 = nodes_data[nid].sum_g2;
            GHPair p_missing_gh = missing_gh_data[pid];
//            float_type p_missing_g2 = missing_g2[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
//            float_type rch_g2 = hist_g2[i];

            auto lch_gh = parent_gh - rch_gh;
//            auto lch_g2 = parent_g2 - hist_g2[i];
            int n_remove = static_cast<int>(param.remove_ratio * n_instances);
            DeltaTree::DeltaGain default_to_left_gain(lch_gh.g, lch_gh.h, rch_gh.g, rch_gh.h, parent_gh.g, parent_gh.h,
                                           p_missing_gh.g, p_missing_gh.h, param.lambda, n_instances, n_remove);
            default_to_left_gain.gain_value = default_to_left_gain.cal_gain_value(mcw);
//            default_to_left_gain.ev_remain_gain = default_to_left_gain.cal_ev_remain_gain(mcw);

            auto default_to_right_gain = DeltaTree::DeltaGain(default_to_left_gain);
            default_to_right_gain.rch_g += p_missing_gh.g;
            default_to_right_gain.rch_h += p_missing_gh.h;
//            default_to_right_gain.rch_g2 += p_missing_g2;
            default_to_right_gain.lch_g -= p_missing_gh.g;
            default_to_right_gain.lch_h -= p_missing_gh.h;
//            default_to_right_gain.lch_g2 -= p_missing_g2;
            default_to_right_gain.gain_value = -default_to_right_gain.cal_gain_value(mcw);
//            default_to_right_gain.ev_remain_gain = default_to_right_gain.cal_ev_remain_gain(mcw);

            if (ft_ge(std::fabs(default_to_left_gain.gain_value), std::fabs(default_to_right_gain.gain_value), 1e-2)) {
                gain[i] = default_to_left_gain;
            }
            else {
                gain[i] = default_to_right_gain;
            }
        }
    }
}

[[deprecated]]
void DeltaTreeBuilder::get_potential_split_points(const vector<vector<gain_pair>> &candidate_idx_gain,
                                                  const int n_nodes_in_level,
                                                  const int *hist_fid, SyncArray<GHPair> &missing_gh,
                                                  SyncArray<GHPair> &hist, vector<float_type> &hist_g2, int level) {
    /**
     * Update sp and tree.nodes, ins2nodes_indices should also be updated by duplication
     * sp should always keep the same size as nodes
    */
    const int nid_offset = num_nodes_per_level.empty() ? 0 :
            std::accumulate(num_nodes_per_level.begin(), num_nodes_per_level.end(), 0);
    const auto hist_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    const auto cut_val_data = cut.cut_points_val.data();
    const auto cut_col_ptr_data = cut.cut_col_ptr.data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    vector<DeltaTree::DeltaNode> updated_nodes;
    const int n_bins = cut.cut_points_val.size();

    assert(nid_offset + candidate_idx_gain.size() == tree.nodes.size());
    int child_offset = 0;       // number of current child nodes
    int potential_offset = 0;   // number of current potential nodes
    bool is_last_layer = (level == param.depth - 1);
    /*
     * nodes from previous levels
     * [nid_offset]
     * prior nodes in this level (with the best gain among children)
     * [potential_offset]
     * potential nodes with these prior nodes
     * [child offset]
     * child nodes of both prior and potential nodes
     */
    vector<GHPair> old_last_hist = last_hist.to_vec();
    parent_indices.clear();

    for (int i = 0; i < candidate_idx_gain.size(); ++i) {

        int num_potential_nodes = candidate_idx_gain[i].size();

        vector<GHPair> base_hist;
        for (int j = 0; j < num_potential_nodes; ++j) {
            int idx_in_level;
            if (j > 0) {    // not prior node
                // duplicate potential nodes
                DeltaTree::DeltaNode node(tree.nodes[nid_offset + i]);
                node.lch_index = nid_offset + n_nodes_in_level + child_offset;     // index of future children (not allocated yet)
                node.rch_index = nid_offset + n_nodes_in_level + child_offset + 1;
                node.final_id = tree.nodes.size();
                tree.nodes.emplace_back(node);
                is_prior.push_back(false);  // not prior node
                idx_in_level = node.final_id - nid_offset;

                // update last_hist (initially, last_hist should be of size n_nodes_in_a_level, now it should be expanded to
                // updated_n_nodes_in_a_level in the end)
                old_last_hist.insert(old_last_hist.end(), base_hist.begin(), base_hist.end());

                // update the list of potential nodes in prior node
                tree.nodes[nid_offset + i].potential_nodes_indices.emplace_back(node.final_id);
            } else {    // prior node
                tree.nodes[nid_offset + i].lch_index = nid_offset + n_nodes_in_level + child_offset;
                tree.nodes[nid_offset + i].rch_index = nid_offset + n_nodes_in_level + child_offset + 1;
                idx_in_level = i;

                base_hist = {old_last_hist.begin() + idx_in_level * n_bins,
                             old_last_hist.begin() + (idx_in_level + 1) * n_bins};

//                tree.nodes[nid_offset + i].potential_nodes_indices.emplace_back(nid_offset + i);
            }

            gain_pair bst = candidate_idx_gain[i][j];
            auto& best_split_gain = bst.second;
            int split_index = bst.first;

            if (!tree.nodes[nid_offset + i].is_valid) {
                sp_data[idx_in_level].split_fea_id = -1;
                sp_data[idx_in_level].nid = -1;
                child_offset += 2;
                parent_indices.push_back(idx_in_level);  // the relative index parent of these two children in layer
                parent_indices.push_back(idx_in_level);

                // todo: check, ThunderGBM uses return;
                continue;
            }
            int fid = hist_fid[split_index];
            sp_data[idx_in_level].split_fea_id = fid;
            sp_data[idx_in_level].nid = idx_in_level + nid_offset;
            sp_data[idx_in_level].gain = best_split_gain;
            size_t n_column = sorted_dataset.n_features();
            sp_data[idx_in_level].fval = cut_val_data[split_index % n_bins];
            sp_data[idx_in_level].split_bid = (split_index % n_bins - cut_col_ptr_data[fid]);
            sp_data[idx_in_level].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
            sp_data[idx_in_level].default_right = best_split_gain.gain_value < 0;
            sp_data[idx_in_level].rch_sum_gh = hist_data[split_index];
            sp_data[idx_in_level].rch_sum_g2 = hist_g2[split_index];
            sp_data[idx_in_level].no_split_value_update = 0;

            child_offset += 2;
            parent_indices.push_back(idx_in_level);  // the relative index parent of these two children in layer
            parent_indices.push_back(idx_in_level);
        }

        // broadcast potential_node_indices
        for (int j = 1;  j < num_potential_nodes; ++j) {
            int potential_node_id = tree.nodes[nid_offset + i].potential_nodes_indices[j];
            tree.nodes[potential_node_id].potential_nodes_indices = tree.nodes[nid_offset + i].potential_nodes_indices;
        }

        // update ins2node_indices
        for (int j = 0; j < n_instances; ++j) {
            auto it = std::find(ins2node_indices[j].begin(), ins2node_indices[j].end(), nid_offset + i);
            if (it != ins2node_indices[j].end()) {
                // substitute each node to its potential nodes
                int idx = it - ins2node_indices[j].begin();
                ins2node_indices[j].erase(it);
                ins2node_indices[j].insert(ins2node_indices[j].begin() + idx,
                                           tree.nodes[nid_offset + i].potential_nodes_indices.begin(),
                                           tree.nodes[nid_offset + i].potential_nodes_indices.end());
            }
        }
    }

    last_hist.load_from_vec(old_last_hist);

    // insert placeholders for children
    vector<DeltaTree::DeltaNode> child_nodes(2 * n_nodes_in_level);
    vector<bool> prior_flags(2 * n_nodes_in_level);
    for (int i = 0; i < child_nodes.size(); ++i) {
        size_t parent_index = child_nodes[i].parent_index = parent_indices[i] + nid_offset;
        child_nodes[i].gain = *(new DeltaTree::DeltaGain());
        child_nodes[i].final_id = nid_offset + n_nodes_in_level + i;
        child_nodes[i].is_leaf = is_last_layer;
        child_nodes[i].potential_nodes_indices = {nid_offset + n_nodes_in_level + i};
        child_nodes[i].lch_index = -3;
        child_nodes[i].rch_index = -3;

        prior_flags[i] = is_prior[parent_index];
    }
    tree.nodes.insert(tree.nodes.end(), child_nodes.begin(), child_nodes.end());
    is_prior.insert(is_prior.end(), prior_flags.begin(), prior_flags.end());
//    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}


void DeltaTreeBuilder::get_split_points(vector<DeltaTree::SplitNeighborhood> &best_split_nbr, int n_nodes_in_level, int *hist_fid,
                                        SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist,
                                        vector<float_type> &hist_g2,  vector<float_type> &missing_g2, int level) {

    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    auto hist_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto &nodes_data = tree.nodes;

    parent_indices.clear();
    parent_indices = vector<int>(2 * n_nodes_in_level);

    auto cut_col_ptr_data = cut.cut_col_ptr.data();
    int n_bins = static_cast<int>(cut.cut_points_val.size());
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        auto &split_nbr = best_split_nbr[i];
        DeltaTree::DeltaGain best_split_gain = split_nbr.best_gain();
        int split_index = split_nbr.best_bid() + n_bins * i;
        parent_indices[2*i] = parent_indices[2*i+1] = i;
        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            continue;
        }
        nodes_data[i + nid_offset].lch_index = nid_offset + n_nodes_in_level + i * 2;
        nodes_data[i + nid_offset].rch_index = nid_offset + n_nodes_in_level + i * 2 + 1;

//        int fid = hist_fid[split_index];
        int fid = split_nbr.fid;
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = best_split_gain;
        int n_column = sorted_dataset.n_features();
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (int) (split_index % n_bins - cut_col_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].fea_missing_g2 = missing_g2[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain.gain_value < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
        sp_data[i].rch_sum_g2 = hist_g2[split_index];
        sp_data[i].no_split_value_update = 0;

        // update split neighbors' bid from index in node to index in the feature
        for (auto &bid: split_nbr.split_bids) {
            bid -= cut_col_ptr_data[fid];
        }
        sp_data[i].split_nbr = split_nbr;
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;

    bool is_last_layer = (level == param.depth - 1);

    // insert placeholders for children
    vector<DeltaTree::DeltaNode> child_nodes(2 * n_nodes_in_level);
    vector<bool> prior_flags(2 * n_nodes_in_level, true);
    for (int i = 0; i < child_nodes.size(); ++i) {
        size_t parent_index = child_nodes[i].parent_index = parent_indices[i] + nid_offset;
        child_nodes[i].gain = DeltaTree::DeltaGain();
        child_nodes[i].final_id = nid_offset + n_nodes_in_level + i;
        child_nodes[i].is_leaf = is_last_layer;
        child_nodes[i].potential_nodes_indices = {nid_offset + n_nodes_in_level + i};
        child_nodes[i].lch_index = -3;
        child_nodes[i].rch_index = -3;
    }
    tree.nodes.insert(tree.nodes.end(), child_nodes.begin(), child_nodes.end());
    is_prior.insert(is_prior.end(), prior_flags.begin(), prior_flags.end());
}




void DeltaTreeBuilder::get_split_points(vector<gain_pair> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                                       SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) {
//    TIMED_SCOPE(timerObj, "get split points");
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    auto hist_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto &nodes_data = tree.nodes;

    auto cut_col_ptr_data = cut.cut_col_ptr.data();
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        const gain_pair& bst = best_idx_gain[i];
        DeltaTree::DeltaGain best_split_gain = bst.second;
        int split_index = bst.first;
        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            // todo: check, ThunderGBM uses return;
            continue;
        }
        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = best_split_gain;
        int n_bins = cut.cut_points_val.size();
        int n_column = sorted_dataset.n_features();
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (int) (split_index % n_bins - cut_col_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain.gain_value < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
        sp_data[i].no_split_value_update = 0;
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}


void DeltaTreeBuilder::get_bin_ids() {
//    SparseColumns &columns = shards[device_id].columns;
//    auto &cut = this->cut;
//    auto &dense_bin_id = this->dense_bin_id;
    using namespace thrust;
    int n_column = sorted_dataset.n_features();
    int nnz = sorted_dataset.csc_val.size();
    auto cut_col_ptr = cut.cut_col_ptr.data();
    auto cut_points_ptr = cut.cut_points_val.data();
    auto csc_val_data = &(sorted_dataset.csc_val[0]);
    auto csc_col_ptr_data = &(sorted_dataset.csc_col_ptr[0]);

    SyncArray<int> bin_id;
    bin_id.resize(nnz);
    auto bin_id_data = bin_id.host_data();
    int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);
    {
        auto lowerBound = [=]__host__(const float_type *search_begin, const float_type *search_end, float_type val) {
            const float_type *left = search_begin;
            const float_type *right = search_end - 1;

            while (left != right) {
                const float_type *mid = left + (right - left) / 2;  // to prevent overflow
                if (*mid <= val)
//                if (ft_le(*mid, val))       // use approximate equality of float_type
                    right = mid;
                else left = mid + 1;
            }
            return left;        // the result satisfies *(left - 1) > val >= *left
        };
        TIMED_SCOPE(timerObj, "binning");

#pragma omp parallel for
        for (int cid = 0; cid < n_column; cid++) {
            for (int i = csc_col_ptr_data[cid]; i < csc_col_ptr_data[cid + 1]; i++) {
                auto search_begin = cut_points_ptr + cut_col_ptr[cid];
                auto search_end = cut_points_ptr + cut_col_ptr[cid + 1];
                auto val = csc_val_data[i];
                bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            }
        }
    }

//    auto max_num_bin = param.max_num_bin;
    dense_bin_id.resize(n_all_instances * n_column);
    auto dense_bin_id_data = dense_bin_id.host_data();
    auto csc_row_idx_data = sorted_dataset.csc_row_idx.data();
    // default bid for missing values
#pragma omp parallel for
    for (int i = 0; i < n_all_instances * n_column; i++) {
        dense_bin_id_data[i] = -1;
    }
#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = csc_col_ptr_data[fid]; i < csc_col_ptr_data[fid + 1]; i++) {
            int row = csc_row_idx_data[i];
            int original_row = -1;
            if (!sorted_dataset.original_indices.empty())
                original_row = sorted_dataset.original_indices[row];    // it is a subset of the original dataset
            else
                original_row = row;
            auto bid = bin_id_data[i];
            dense_bin_id_data[original_row * n_column + fid] = bid;
        }
    }
//    LOG(DEBUG);
}

void DeltaTreeBuilder::update_random_feature_rank_(size_t seed) {
    std::mt19937 rng{seed};
    random_feature_rank.resize(sorted_dataset.n_features());
    std::iota(random_feature_rank.begin(), random_feature_rank.end(), 0);
    std::shuffle(random_feature_rank.begin(), random_feature_rank.end(), rng);
}

void DeltaTreeBuilder::update_random_split_nbr_rank_(size_t seed) {
    std::mt19937 rng{seed};
    int n_split_nbrs = 0;
    for (int i = 0; i < cut.cut_col_ptr.size() - 1; ++i) {
        int start = cut.cut_col_ptr[i];
        int end = cut.cut_col_ptr[i+1];
        n_split_nbrs += std::max(end - param.nbr_size + 1 - start, 1);
    }
    random_split_nbr_rank.resize(n_split_nbrs);
    std::iota(random_split_nbr_rank.begin(), random_split_nbr_rank.end(), 0);
    std::shuffle(random_split_nbr_rank.begin(), random_split_nbr_rank.end(), rng);
}

void DeltaTreeBuilder::update_indices_in_split_nbr(vector<DeltaTree::SplitNeighborhood> &split_nbrs, const vector<int>& node_indices) {
#pragma omp parallel for
    for (int i = 0; i < split_nbrs.size(); ++i) {
        auto &split_nbr = split_nbrs[i];
        int node_id = node_indices[i];
        split_nbr.marginal_indices.resize(split_nbr.split_bids.size());
#pragma omp parallel for
        for (int j = 0; j < split_nbr.split_bids.size(); ++j) {
            int bid = split_nbr.split_bids[j];
            int feature_offset = cut.cut_col_ptr[split_nbr.fid];
            split_nbr.marginal_indices[j] = std::unordered_set<int>();
            const auto &all_marginal_indices = cut.indices_in_hist[split_nbr.fid][bid - feature_offset];
            // copy all_marginal_indices in this node to split_nbr.marginal_indices[j]
            for (int id: all_marginal_indices) {
                if (ins2node_indices[id][0] == node_id) {   // node_id > 0, thus ins2node_indices[id][0] > 0
                    split_nbr.marginal_indices[j].insert(id);
                }
            }
//            std::copy_if(all_marginal_indices.begin(), all_marginal_indices.end(),
//                         std::inserter(split_nbr.marginal_indices[j], split_nbr.marginal_indices[j].begin()),
//                         [&](int id){
//                             return ins2node_indices[id][0] == node_id;
//            });
//            LOG(DEBUG);
//            split_nbr.marginal_indices[j] = cut.indices_in_hist[split_nbr.fid][bid - feature_offset];
        }
    }
}







