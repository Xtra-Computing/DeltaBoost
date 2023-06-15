//
// Created by HUSTW on 8/17/2021.
//

#ifndef FEDTREE_DELTA_TREE_REMOVER_H
#define FEDTREE_DELTA_TREE_REMOVER_H


#include <memory>
#include <utility>
#include <FedTree/dataset.h>
#include <unordered_map>

#include "FedTree/Tree/tree.h"


class DeltaTreeRemover {
public:
    DeltaTreeRemover() = default;

    DeltaTreeRemover(DeltaTree* tree_ptr, const DataSet* dataSet, DeltaBoostParam param, vector<GHPair> gh_pairs,
                     vector<vector<int>> ins2node_indices, const vector<bool>& is_subset_indices):
    tree_ptr(tree_ptr), dataSet(dataSet), param(std::move(param)), gh_pairs(std::move(gh_pairs)),
    ins2node_indices(std::move(ins2node_indices)), is_subset_indices(is_subset_indices) { }

    DeltaTreeRemover(DeltaTree *tree_ptr, const DataSet *dataSet, const DeltaBoostParam &param, const vector<bool>& is_subset_indices) :
    tree_ptr(tree_ptr), dataSet(dataSet), param(param), is_subset_indices(is_subset_indices) {
        gh_pairs = std::vector<GHPair>(dataSet->n_instances());
        ins2node_indices = std::vector<std::vector<int>>(
                dataSet->n_instances(), std::vector<int>(0));
    }

    DeltaTreeRemover(DeltaTree *tree_ptr, const DataSet *dataSet, const DeltaBoostParam &param) :
            tree_ptr(tree_ptr), dataSet(dataSet), param(param){
        gh_pairs = std::vector<GHPair>(dataSet->n_instances());
        ins2node_indices = std::vector<std::vector<int>>(
                dataSet->n_instances(), std::vector<int>(0));
    }

    void remove_sample_by_id(int id);
    void adjust_gradients_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs);
    void adjust_split_nbrs_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs, bool remove_n_ins=false);
    void remove_samples_by_indices(const vector<int>& indices);
    void prune();
    void sort_potential_nodes_by_gain(int root_idx);

    void get_invalid_sp(const vector<int> &dense_bin_id, const RobustHistCut& cut, const vector<int>& removed_indices,
                        vector<vector<int>> &invalid_bids);
    void get_invalid_sp(DeltaCut &cut, const DataSet &dataset, const vector<int>& removed_indices, int max_bin_size);

    DeltaTree *tree_ptr = nullptr;
    DeltaBoostParam param;
    vector<GHPair> gh_pairs;
    vector<vector<int>> ins2node_indices;
    vector<bool> is_subset_indices;
    vector<bool> is_iid_removed;
    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int>>> invalid_bins;

    const DataSet* dataSet = nullptr;   // this dataset should be the original dataset instead of the sorted_dataset
};

/**
 *
 */
template<class It>
struct remover
{
    size_t *i;
    It *begin;
    It const *end;
    explicit remover(size_t &i, It &begin, It const &end) : i(&i), begin(&begin), end(&end) { }
    template<class T>
    bool operator()(T const &)
    {
        size_t &i = *this->i;
        It &begin = *this->begin;
        It const &end = *this->end;
        while (begin != end && *begin < i)  /* only necessary in case there are duplicate indices */
        { ++begin;  }
        bool const b = begin != end && *begin == i;
        if (b) { ++begin; }
        ++i;
        return b;
    }
};
template<class Container, class IndexIt>
IndexIt remove_indices(Container &items, IndexIt indices_begin, IndexIt const &indices_end)
{
    size_t i = 0;
    std::sort(indices_begin, indices_end);
    items.erase(std::remove_if(items.begin(), items.end(), remover<IndexIt>(i, indices_begin, indices_end)), items.end());
    return indices_begin;
}

template<typename T>
auto clean_vectors_by_indices_(vector<vector<T>>& vectors, const vector<int> &indices) {
    vectors.erase(std::remove_if(vectors.begin(), vectors.end(), [&](vector<T> &vec){
        int index = &vec - &*vectors.begin();
        return indices[index] == -1;
    }), vectors.end());
}

#endif //FEDTREE_DELTA_TREE_REMOVER_H
