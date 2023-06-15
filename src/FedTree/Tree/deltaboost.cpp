#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/Tree/deltaboost.h"
#include "FedTree/booster.h"
#include "FedTree/deltabooster.h"
#include "FedTree/Tree/delta_tree_remover.h"
#include "FedTree/Tree/deltaboost_remover.h"

void DeltaBoost::train(DeltaBoostParam &param, DataSet &dataset) {
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist") {
        std::cout << "FedTree only supports histogram-based training yet";
        exit(1);
    }

    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_round = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

//    std::map<int, vector<int>> batch_idxs;
//    Partition partition;
//    vector<DataSet> subsets(3);
//    partition.homo_partition(dataset, 3, true, subsets, batch_idxs);
//
    LOG(INFO) << "starting building trees";
    std::chrono::high_resolution_clock timer;

    DeltaBooster booster;
    // record time of init
    auto start_init = timer.now();
//    if (param.hash_sampling_round > 1)
//        booster.init(dataset, param, dataset.n_instances(), true, true);
//    else
//        booster.init(dataset, param, dataset.n_instances(), true, false);
    booster.init(dataset, param, dataset.n_instances());
    auto end_init = timer.now();
    std::chrono::duration<float> init_time = end_init - start_init;
    LOG(INFO) << "Init booster time: " << init_time.count() << "s";

    dataset.set_seed((int)param.seed);
    dataset.get_row_hash_();

    auto start_train = timer.now();
    for (int i = 0; i < param.n_trees; ++i) {
        // subsampling by hashing
        if (param.hash_sampling_round > 1) {
            if (i % param.hash_sampling_round == 0) {
                dataset.set_seed((int)param.seed + i);
                dataset.get_row_hash_();
                dataset.update_sampling_by_hashing_(param.hash_sampling_round);
            }
            auto &sub_dataset = dataset.get_sampled_dataset(i % param.hash_sampling_round);
//            auto &sub_dataset = dataset.get_sampled_dataset(0);

            booster.reset(sub_dataset, param);

            if (i > 0) {
                predict_raw(param, sub_dataset, booster.fbuilder->y_predict);
            }

            auto subset_indices = dataset.get_subset_indices(i % param.hash_sampling_round);
//            auto subset_indices = dataset.get_subset_indices(0);
            is_subset_indices_in_tree.emplace_back(indices_to_hash_table(subset_indices, dataset.n_instances()));
        } else {
            is_subset_indices_in_tree.emplace_back(vector<bool>());
        }

        //one iteration may produce multiple trees, depending on objectives
        booster.boost(trees, gh_pairs_per_sample, ins2node_indices_per_tree, dataset.row_hash, is_subset_indices_in_tree[i]);

//        SyncArray<float_type> y_predict_tmp;
//        predict_raw(param, booster.fbuilder->sorted_dataset, y_predict_tmp);
        int valid_size = 0;
        for (const auto &node: trees[trees.size() - 1][0].nodes) {
            if (node.is_valid) ++valid_size;
        }
        LOG(DEBUG) << "Tree " << i << ", Number of nodes:" << valid_size;
    }

//    float_type score = predict_score(param, dataset);
//    LOG(INFO) << score;

    auto stop_train = timer.now();
    std::chrono::duration<float> training_time = stop_train - start_train;
    LOG(INFO) << "training time = " << training_time.count();

    return;
}


//void DeltaBoost::remove_samples(DeltaBoostParam &param, DataSet &dataset, const vector<int>& sample_indices) {
//    typedef std::chrono::high_resolution_clock timer;
//    auto start_time = timer::now();
//    auto end_time = timer::now();
//    std::chrono::duration<double> duration = end_time - start_time;
//
//    LOG(INFO) << "start removing samples";
//
//    start_time = timer::now();
//
//    SyncArray<float_type> y = SyncArray<float_type>(dataset.n_instances());
//    y.copy_from(dataset.y.data(), dataset.n_instances());
//    std::unique_ptr<ObjectiveFunction> obj(ObjectiveFunction::create(param.objective));
//    obj->configure(param, dataset);     // slicing param
//
//    LOG(INFO) << "Preparing for deletion";
//
//    DeltaBoostRemover deltaboost_remover;
//    if (param.hash_sampling_round > 1) {
//        deltaboost_remover = DeltaBoostRemover(&dataset, &trees, is_subset_indices_in_tree, obj.get(), param);
//    } else {
//        start_time = timer::now();
//
//        deltaboost_remover = DeltaBoostRemover(&dataset, &trees, obj.get(), param);
//
//        end_time = timer::now();
//        duration = end_time - start_time;
//        LOG(DEBUG) << "[Removing time] Step 0 (out) = " << duration.count();
//    }
//
//    deltaboost_remover.n_all_instances = dataset.n_instances();
//
////    deltaboost_remover.get_info_by_prediction(gh_pairs_per_sample);
//    deltaboost_remover.get_info(gh_pairs_per_sample, ins2node_indices_per_tree);
//
//    LOG(INFO) << "Deleting " << param.n_used_trees << " trees";
//
//#pragma omp parallel for
//    for (int i = 0; i < param.n_used_trees; ++i) {
////        DeltaTree &tree = trees[i][0];
////        vector<GHPair>& gh_pairs = gh_pairs_per_sample[i];
////        auto &ins2node_indices = ins2node_indices_per_tree[i];
////        DeltaTreeRemover tree_remover(&tree, &dataset, param, gh_pairs, ins2node_indices);
//
//        DeltaTreeRemover& tree_remover = deltaboost_remover.tree_removers[i];
//        vector<bool> is_iid_removed = indices_to_hash_table(sample_indices, dataset.n_instances());
//        tree_remover.is_iid_removed = is_iid_removed;
//        const std::vector<GHPair>& gh_pairs = tree_remover.gh_pairs;
//        vector<int> trained_sample_indices;
//        if (param.hash_sampling_round > 1) {
//            std::copy_if(sample_indices.begin(), sample_indices.end(), std::back_inserter(trained_sample_indices), [&](int idx){
//                return is_subset_indices_in_tree[i][idx];
//            });
//        } else {
//            trained_sample_indices = sample_indices;
//        }
//
//        tree_remover.remove_samples_by_indices(trained_sample_indices);
//        tree_remover.prune();
//
////        if (i > 0) {
////            SyncArray<float_type> y_predict;
////            predict_raw(param, dataset, y_predict, i);
////
////            SyncArray<GHPair> updated_gh_pairs_array(y.size());
////            obj->get_gradient(y, y_predict, updated_gh_pairs_array);
////            vector<GHPair> delta_gh_pairs = updated_gh_pairs_array.to_vec();
////            auto quantized_gh_pairs = DeltaBooster::quantize_gradients(delta_gh_pairs, param.n_quantize_bins, dataset.row_hash);
//////            GHPair sum_gh_pair = std::accumulate(delta_gh_pairs.begin(), delta_gh_pairs.end(), GHPair());
////
////            vector<int> adjust_indices;
////            vector<GHPair> adjust_values;
////            for (int j = 0; j < quantized_gh_pairs.size(); ++j) {
////                if (is_iid_removed[j] || (param.hash_sampling_round > 1 && !is_subset_indices_in_tree[i][j])) continue;
//////                if (std::fabs(quantized_gh_pairs[j].g - gh_pairs[j].g) > 1e-6 ||
//////                    std::fabs(quantized_gh_pairs[j].h - gh_pairs[j].h) > 1e-6) {
////                    adjust_indices.emplace_back(j);
////                    adjust_values.emplace_back(quantized_gh_pairs[j] - gh_pairs[j]);
//////                }
////            }
//////            GHPair sum_delta_gh_pair = std::accumulate(adjust_values.begin(), adjust_values.end(), GHPair());
////
//////            // debug only
//////            SyncArray<int> adjust_indices_array;
//////            SyncArray<GHPair> adjust_values_array;
//////            adjust_indices_array.load_from_vec(adjust_indices);
//////            adjust_values_array.load_from_vec(adjust_values);
//////            LOG(DEBUG) << "Adjusted indices" << adjust_indices_array;
//////            LOG(DEBUG) << "Adjusted values" << adjust_values_array;
////
////            tree_remover.adjust_split_nbrs_by_indices(adjust_indices, adjust_values, false);
////        }
//    }
//
//    end_time = timer::now();
//    duration = end_time - start_time;
//    LOG(INFO) << "Removing time in function = " << duration.count();
//}

float_type DeltaBoost::predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees) {
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict, num_trees);
    LOG(DEBUG) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    float_type score = metric->get_score(y_predict);

//    LOG(INFO) << metric->get_name().c_str() << " = " << score;
    LOG(INFO) << "Test: " << metric->get_name() << " = " << score;
    return score;
}

float_type DeltaBoost::predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, vector<float_type> &raw_predict, int num_trees) {
    /**
     * @brief predict scores and return raw predictions in raw_predict
     */
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict, num_trees);
    LOG(DEBUG) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    float_type score = metric->get_score(y_predict);

    raw_predict = y_predict.to_vec();

    LOG(INFO) << "Test: " << metric->get_name() << " = " << score;
    return score;
}

void DeltaBoost::predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict,
                             int num_trees) {
    // measure time
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();

    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = num_trees == -1 ? trees.size() : num_trees;
    int num_class = trees.front().size();

//    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = dataSet.csr_col_idx.data();
    auto csr_val_data = dataSet.csr_val.data();
    auto csr_row_ptr_data = dataSet.csr_row_ptr.data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    //use sparse format and binary search
#pragma omp parallel for  // remove for debug
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](const DeltaTree::DeltaNode& node, float_type feaValue) {
            return feaValue < node.split_value ? node.lch_index : node.rch_index;
//            return ft_ge(feaValue, node.split_value) ? node.rch_index : node.lch_index;
        };
        auto get_val_dense = [](const int *row_idx, const float_type *row_val, int idx) -> float_type {
            assert(idx == row_idx[idx]);
            return row_val[idx];
        };
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
//                const DeltaTree::DeltaNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
//                DeltaTree::DeltaNode curNode = node_data[0];
//                DeltaTree::DeltaNode *cur_node = trees[iter][t].nodes[0];
                const auto & node_data = trees[iter][t].nodes;
                const DeltaTree::DeltaNode* cur_node_ptr = &node_data[0];
                int cur_nid = 0; //node id
                int depth = 0;
                int last_idx = -1;
                while (!cur_node_ptr->is_leaf) {
                    if (cur_node_ptr->lch_index < 0 || cur_node_ptr->rch_index < 0) {
                        LOG(FATAL);
                    }
                    int fid = cur_node_ptr->split_feature_id;
                    bool is_missing = false;
//                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    float_type fval = get_val_dense(col_idx, row_val, fid);
                    if (!is_missing)
                        cur_nid = get_next_child(*cur_node_ptr, fval);
                    else if (cur_node_ptr->default_right)
                        cur_nid = cur_node_ptr->rch_index;
                    else
                        cur_nid = cur_node_ptr->lch_index;
//                    const auto& cur_potential_node = node_data[cur_nid];
//                    int cur_node_idx = cur_potential_node.potential_nodes_indices[0];
                    cur_node_ptr = &node_data[cur_nid];
                    depth++;
                }
                sum += lr * node_data[cur_nid].base_weight;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }

    auto end_predict = clock.now();
    std::chrono::duration<double> diff = end_predict - start;
    LOG(DEBUG) << "predict time: " << diff.count() << "s";
}

vector<float_type> DeltaBoost::predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees) {
    /**
     * This function is a wrapper for predict_raw with SyncArray. The return value is expected to be used for initialization
     * instead of assignment. E.g., auto y_predict = predict_raw(param, ...); In this way, the copying would be optimized
     * by "named return value optimization (NRVO)".
     */
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict, num_trees);
    return y_predict.to_vec();
}

void DeltaBoost::trim_unused_members_() {
    // In each tree.cut, remove indices in each node
    for (auto & tree : trees) {
        auto & cut = tree[0].cut;
        for (auto & bin_tree: cut.bin_trees) {
            for (auto & bin : bin_tree.bins) {
                bin.indices.clear();
            }
        }
        cut.n_instances_in_hist.clear();
        cut.indices_in_hist.clear();
    }

}



#pragma clang diagnostic pop