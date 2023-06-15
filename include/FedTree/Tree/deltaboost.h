//
// Created by HUSTW on 7/31/2021.
//



#ifndef FEDTREE_DELTABOOST_H
#define FEDTREE_DELTABOOST_H


#include "gbdt.h"
#include "../deltabooster.h"
#include "delta_tree_remover.h"
#include "deltaboost_remover.h"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/vector.hpp"


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

//    inline __attribute__((always_inline)) void remove_samples(DeltaBoostParam &param, DataSet &dataset, const vector<int>& sample_indices) {
//
//    }
    void trim_unused_members_();

    DeltaBoost slim_down();

private:
    friend class boost::serialization::access;

    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar & trees;
        ar & gh_pairs_per_sample;
        ar & is_subset_indices_in_tree;
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

        // temporarily convert vector bool to int because vector bool is not support by boost::json 1.75
        auto is_subset_indices_in_tree_int = json::value_to<std::vector<std::vector<int>>>(v.at("is_subset_indices_in_tree"));
        // concert every element in vector to bool, stored in is_subset_indices_in_tree
        for (int i = 0; i < is_subset_indices_in_tree_int.size(); ++i) {
            vector<bool> temp;
            for (int j = 0; j < is_subset_indices_in_tree_int[i].size(); ++j) {
                temp.push_back(is_subset_indices_in_tree_int[i][j] == 1);
            }
            deltaBoost.is_subset_indices_in_tree.push_back(temp);
        }
        return deltaBoost;
    }

    //json parser
    friend void tag_invoke(json::value_from_tag, json::value& v, DeltaBoost const& deltaBoost) {
        // temporarily convert vector bool to int because vector bool is not support by boost::json 1.75
        vector<vector<int>> is_subset_indices_in_tree_int;
        for (int i = 0; i < deltaBoost.is_subset_indices_in_tree.size(); ++i) {
            vector<int> temp;
            for (int j = 0; j < deltaBoost.is_subset_indices_in_tree[i].size(); ++j) {
                temp.push_back(deltaBoost.is_subset_indices_in_tree[i][j] ? 1 : 0);
            }
            is_subset_indices_in_tree_int.push_back(temp);
        }
        v = json::object {
                {"trees", json::value_from(deltaBoost.trees)},
                {"gh_pairs_per_sample", json::value_from(deltaBoost.gh_pairs_per_sample)},
                {"is_subset_indices_in_tree", json::value_from(is_subset_indices_in_tree_int)},
                {"ins2node_indices_per_tree", json::value_from(deltaBoost.ins2node_indices_per_tree)}
        };
    }

};

#endif //FEDTREE_DELTABOOST_H
