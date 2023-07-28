//
// Created by HUSTW on 7/31/2021.
//

#include "gbdt.h"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/vector.hpp"

#ifndef FEDTREE_DELTABOOST_H
#define FEDTREE_DELTABOOST_H

class DeltaBoost : public GBDT{
public:
    vector<vector<DeltaTree>> trees;

    vector<vector<GHPair>> gh_pairs_per_sample;       // first index is the iteration, second index is the sample ID
    vector<vector<vector<int>>> ins2node_indices_per_tree;  // first index is tree id, second index is sample id, third index is node id

    vector<vector<bool>> is_subset_indices_in_tree;     // whether an instance is trained in a tree. Size [n_trees, n_instances]
                                                        // true means trained, false means not trained.

    DeltaBoost() = default;

    explicit DeltaBoost(const vector<vector<DeltaTree>>& gbdt){
        trees = gbdt;
    }

    void train(DeltaBoostParam &param, DataSet &dataset);

    float_type predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees=-1);

    float_type predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, vector<float_type> &raw_predict, int num_trees=-1);

    void predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict,
                                 int num_trees=-1);

    vector<float_type> predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees=-1);

    void remove_samples(DeltaBoostParam &param, DataSet &dataset, const vector<int>& sample_indices);

private:
    friend class boost::serialization::access;

    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar & trees;
        ar & gh_pairs_per_sample;
        ar & ins2node_indices_per_tree;
    }

    // json parser
    friend DeltaBoost tag_invoke(json::value_to_tag<DeltaBoost>, json::value const& v) {
        auto &o = v.as_object();

        DeltaBoost deltaBoost;

        deltaBoost.trees = json::value_to<std::vector<std::vector<DeltaTree>>>(v.at("trees"));
        deltaBoost.gh_pairs_per_sample = json::value_to<std::vector<std::vector<GHPair>>>(v.at("gh_pairs_per_sample"));
//        deltaBoost.ins2node_indices_per_tree = json::value_to<std::vector<std::vector<std::vector<int>>>>(
//                v.at("ins2node_indices_per_tree"));

        return deltaBoost;
    }

    //json parser
    friend void tag_invoke(json::value_from_tag, json::value& v, DeltaBoost const& deltaBoost) {
        v = json::object {
                {"trees", json::value_from(deltaBoost.trees)},
                {"gh_pairs_per_sample", json::value_from(deltaBoost.gh_pairs_per_sample)},
//                {"ins2node_indices_per_tree", json::value_from(deltaBoost.ins2node_indices_per_tree)}
        };
    }

};

#endif //FEDTREE_DELTABOOST_H
