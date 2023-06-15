//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/deltabooster.h"

#include <memory>
#include "FedTree/Tree/deltaboost.h"


void DeltaBooster::init(DataSet &dataSet, const DeltaBoostParam &delta_param, int n_all_instances, bool get_cut_points,
                        bool skip_get_bin_ids) {
    param = delta_param;

    this->n_all_instances = n_all_instances;
    fbuilder = std::make_unique<DeltaTreeBuilder>();
    fbuilder->n_all_instances = n_all_instances;

    if(get_cut_points)
        fbuilder->init(dataSet, param, skip_get_bin_ids);
    else {
        fbuilder->init_nocutpoints(dataSet, param);
    }


    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}


void DeltaBooster::reset(DataSet &dataSet, const DeltaBoostParam &delta_param, bool get_cut_points) {
    param = delta_param;

//    fbuilder = std::make_unique<DeltaTreeBuilder>();
    if(get_cut_points)
        fbuilder->reset(dataSet, param);    // avoid updating cut, cuz cut should fit the global dataset
    else {
        LOG(FATAL) << "Not supported yet";
        fbuilder->init_nocutpoints(dataSet, param);
    }
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}


void DeltaBooster::boost(vector<vector<DeltaTree>>& boosted_model, vector<vector<GHPair>>& gh_pairs_per_sample,
                         vector<vector<vector<int>>>& ins2node_indices_per_tree, const vector<int> &row_hash,
                         const vector<bool>& is_subset_indices_in_tree) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);

    //update gradients
    SyncArray<GHPair> original_gh(gradients.size());
    obj->get_gradient(y, fbuilder->get_y_predict(), original_gh);

    // get row_hash in tree
    vector<int> row_hash_in_tree;
    if (is_subset_indices_in_tree.empty()) {
        row_hash_in_tree = row_hash;
    } else {
        for (int i = 0; i < row_hash.size(); ++i) {
            if (is_subset_indices_in_tree[i]) {
                row_hash_in_tree.push_back(row_hash[i]);
            }
        }
    }

    // quantize gradients if needed. todo: optimize these per-instance copy
    float_type g_bin_width = 1., h_bin_width = 1.;
    if (param.n_quantize_bins > 0) {
        gradients.load_from_vec(quantize_gradients(original_gh.to_vec(), param.n_quantize_bins, row_hash_in_tree, g_bin_width, h_bin_width));
    } else {
        gradients.copy_from(original_gh);
    }

    assert(g_bin_width > 0 && h_bin_width > 0);
    fbuilder->g_bin_width = g_bin_width;
    fbuilder->h_bin_width = h_bin_width;

    if (param.hash_sampling_round > 1) {
        // map subset gradients to global gradients, filled with 0 if not in subset
        vector<GHPair> all_gradients(n_all_instances, GHPair(0, 0));
        int j = 0;
        for (int i = 0; i < all_gradients.size(); ++i) {
            int tree_idx = (int) boosted_model.size();
            if (is_subset_indices_in_tree[i]) {
                all_gradients[i] = gradients.host_data()[j++];
            }
        }
        assert(j == gradients.size());      // all the gradients must be copied
        gh_pairs_per_sample.push_back(all_gradients);
        gradients.load_from_vec(all_gradients);     // todo: optimize this copy
    } else {
        gh_pairs_per_sample.push_back(gradients.to_vec());
    }

    std::vector<std::vector<int>> ins2node_indices;
//    if (param.bagging) rowSampler.do_bagging(gradients);

    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_delta_approximate(gradients, ins2node_indices, is_subset_indices_in_tree));
    PERFORMANCE_CHECKPOINT(timerObj);
    ins2node_indices_per_tree.push_back(ins2node_indices);

    //show metric on training set
    std::ofstream myfile;
    myfile.open ("data.txt", std::ios_base::app);
    myfile << fbuilder->get_y_predict() << "\n";
    myfile.close();
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}

vector<GHPair> DeltaBooster::quantize_gradients(const vector<GHPair> &gh, int n_bins, const vector<int> &row_hash,
                                                float_type &g_bin_width, float_type &h_bin_width) {
    /**
     * Randomly quantize gradients and hessians to neighboring grids.
     */
    assert(gh.size() == row_hash.size());
    vector<GHPair> quantized_gh(gh.size());

    // get max absolute value of gh.g and gh.h
    float_type max_abs_g = 0, max_abs_h = 0;
    for (int i = 0; i < gh.size(); ++i) {
        max_abs_g = std::max(max_abs_g, std::abs(gh[i].g));
        max_abs_h = std::max(max_abs_h, std::abs(gh[i].h));
    }

    // calculate bin width
    g_bin_width = max_abs_g / n_bins;
    h_bin_width = max_abs_h / (n_bins * 2);      // smaller width according to the NeurIPS-22 paper
    assert(g_bin_width > 0 && h_bin_width > 0);

//    std::mt19937 gen1(seed);
//    std::mt19937 gen2(seed + 1);

    // random round gh to integers (DO NOT run in parallel to ensure random sequence is the same)
    for (int i = 0; i < gh.size(); ++i) {
        quantized_gh[i].g = random_round(gh[i].g / g_bin_width, row_hash[i]);
        quantized_gh[i].h = random_round(gh[i].h / h_bin_width, row_hash[i] + 1);
    }

    auto sum_gh = std::accumulate(quantized_gh.begin(), quantized_gh.end(), GHPair(0, 0));

    return quantized_gh;
}

float_type DeltaBooster::random_round(float_type x, float_type left, float_type right, size_t seed) {
    /*
     * Randomly round x to the left or right. The expected value of the result is x.
     * The probability of rounding to the left is (right - x) / (right - left);
     * The probability of rounding to the right is (x - left) / (right - left);
     */
    float_type prob = (x - left) / (right - left);
    if (prob < 0 || prob > 1) {
        LOG(FATAL) << "prob = " << prob << ", left = " << left << ", right = " << right << ", x = " << x;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1);
    float_type rand = dis(gen);
    if (rand > prob) {
        return std::floor(x);
    } else {
        return std::ceil(x);
    }
}

float_type DeltaBooster::random_round(float_type x, size_t seed) {
    /*
     * Randomly round x to the floor or ceiling integer (of float_type). The expected value of the result is x.
     * The probability of rounding to the ceiling is (x - floor(x));
     * The probability of rounding to the floor is (ceiling(x) - x);
     */

    std::mt19937 gen{seed};
    std::uniform_real_distribution<> dis(0, 1);
    float_type rand = dis(gen);
    if (rand > x - std::floor(x)) {
        return std::floor(x);
    } else {
        return std::ceil(x);
    }
}






