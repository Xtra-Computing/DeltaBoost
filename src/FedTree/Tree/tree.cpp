//
// Created by liqinbin on 10/14/20.
//

#include "FedTree/Tree/tree.h"
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include <numeric>
#include <boost/compute/algorithm/reduce.hpp>

void Tree::init_CPU(const GHPair sum_gh, const GBDTParam &param) {
    init_structure(param.depth);
    float_type lambda = param.lambda;
    auto node_data = nodes.host_data();
    Tree::TreeNode &root_node = node_data[0];
    root_node.sum_gh_pair = sum_gh;
    root_node.is_valid = true;
    root_node.calc_weight_(lambda);
}

void Tree::init_CPU(const SyncArray<GHPair> &gradients, const GBDTParam &param) {
    TIMED_FUNC(timerObj);
    init_structure(param.depth);
    //init root node
    GHPair sum_gh = thrust::reduce(thrust::host, gradients.host_data(), gradients.host_end());
    LOG(DEBUG) << "init_CPU: " << sum_gh;

    float_type lambda = param.lambda;
    auto node_data = nodes.host_data();
    Tree::TreeNode &root_node = node_data[0];
    root_node.sum_gh_pair = sum_gh;
    root_node.is_valid = true;
    root_node.calc_weight_(lambda); // TODO: check here
    root_node.n_instances = gradients.size();
}

void Tree::init_structure(int depth) {
    int n_max_nodes = static_cast<int>(pow(2, depth + 1) - 1);
    nodes = SyncArray<TreeNode>(n_max_nodes);
    auto node_data = nodes.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_max_nodes; i++) {
        node_data[i].final_id = i;
        node_data[i].split_feature_id = -1;
        node_data[i].is_valid = false;
        node_data[i].parent_index = i == 0 ? -1 : (i - 1) / 2;
        node_data[i].n_instances = 0;
        if (i < n_max_nodes / 2) {
            node_data[i].is_leaf = false;
            node_data[i].lch_index = i * 2 + 1;
            node_data[i].rch_index = i * 2 + 2;
        } else {
            //leaf nodes
            node_data[i].is_leaf = true;
            node_data[i].lch_index = -1;
            node_data[i].rch_index = -1;
        }
    }

};

// GPU version of tree initialization
// void Tree::init2(const SyncArray<GHPair> &gradients, const GBMParam &param) {
//     TIMED_FUNC(timerObj);
//     int n_max_nodes = static_cast<int>(pow(2, param.depth + 1) - 1);
//     nodes = SyncArray<TreeNode>(n_max_nodes);
//     auto node_data = nodes.device_data();
//     device_loop(n_max_nodes, [=]__device__(int i) {
//         node_data[i].final_id = i;
//         node_data[i].split_feature_id = -1;
//         node_data[i].is_valid = false;
//         node_data[i].parent_index = i == 0 ? -1 : (i - 1) / 2;
//         if (i < n_max_nodes / 2) {
//             node_data[i].is_leaf = false;
//             node_data[i].lch_index = i * 2 + 1;
//             node_data[i].rch_index = i * 2 + 2;
//         } else {
//             //leaf nodes
//             node_data[i].is_leaf = true;
//             node_data[i].lch_index = -1;
//             node_data[i].rch_index = -1;
//         }
//     });

//     //init root node
//     GHPair sum_gh = thrust::reduce(thrust::cuda::par, gradients.device_data(), gradients.device_end());
//     float_type lambda = param.lambda;
//     device_loop<1, 1>(1, [=]__device__(int i) {
//         Tree::TreeNode &root_node = node_data[0];
//         root_node.sum_gh_pair = sum_gh;
//         root_node.is_valid = true;
//         root_node.calc_weight_(lambda);
//     });
// }

string Tree::dump(int depth) const {
    string s("\n");
    preorder_traversal(0, depth, 0, s);
    return s;
}

void Tree::preorder_traversal(int nid, int max_depth, int depth, string &s) const {
    if (nid == -1)//child of leaf node
        return;
    const TreeNode &node = nodes.host_data()[nid];
    const TreeNode *node_data = nodes.host_data();
    if (node.is_valid && !node.is_pruned) {
        s = s + string(static_cast<unsigned long>(depth), '\t');

        if (node.is_leaf) {
            s = s + string_format("%d:leaf=%.6g\n", node.final_id, node.base_weight);
        } else {
            int lch_final_id = node_data[node.lch_index].final_id;
            int rch_final_id = node_data[node.rch_index].final_id;
            string str_inter_node = string_format("%d:[f%d<%.6g] yes=%d,no=%d,missing=%d\n", node.final_id,
                                                  node.split_feature_id + 1,
                                                  node.split_value, lch_final_id, rch_final_id,
                                                  node.default_right == 0 ? lch_final_id : rch_final_id);
            s = s + str_inter_node;
        }
//             string_format("%d:[f%d<%.6g], weight=%f, gain=%f, dr=%d\n", node.final_id, node.split_feature_id + 1,
//                           node.split_value,
//                           node.base_weight, node.gain, node.default_right));
    }
    if (depth < max_depth) {
        preorder_traversal(node.lch_index, max_depth, depth + 1, s);
        preorder_traversal(node.rch_index, max_depth, depth + 1, s);
    }
}

std::ostream &operator<<(std::ostream &os, const Tree::TreeNode &node) {
    os << string_format("\nnid:%d,l:%d,v:%d,p:%d,lch:%d,rch:%d,split_feature_id:%d,f:%f,gain:%f,r:%d,w:%f,",
                        node.final_id, node.is_leaf,
                        node.is_valid, node.is_pruned, node.lch_index, node.rch_index,
                        node.split_feature_id, node.split_value, node.gain, node.default_right, node.base_weight);
    os << "g/h:" << node.sum_gh_pair;
    return os;
}

void Tree::reorder_nid() {
    int nid = 0;
    Tree::TreeNode *nodes_data = nodes.host_data();
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes_data[i].is_valid && !nodes_data[i].is_pruned) {
            nodes_data[i].final_id = nid;
            nid++;
        }
    }
}

/*
void Tree::compute_leaf_value() {
    Tree::TreeNode *nodes_data = this->nodes.host_data();
    for(int i = 0; i < this->nodes.size(); i++) {
        if(nodes_data[i].is_leaf) {
            nodes_data[i].calc_weight_();
        }
    }
}
*/

int Tree::try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count) {
    Tree::TreeNode *nodes_data = nodes.host_data();
    int p_nid = nodes_data[nid].parent_index;
    if (p_nid == -1) return np;// is root
    Tree::TreeNode &p_node = nodes_data[p_nid];
    Tree::TreeNode &lch = nodes_data[p_node.lch_index];
    Tree::TreeNode &rch = nodes_data[p_node.rch_index];
    leaf_child_count[p_nid]++;
    if (leaf_child_count[p_nid] >= 2 && p_node.gain < gamma) {
        //do pruning
        //delete two children
        CHECK(lch.is_leaf);
        CHECK(rch.is_leaf);
        lch.is_pruned = true;
        rch.is_pruned = true;
        //make parent to leaf
        p_node.is_leaf = true;
        return try_prune_leaf(p_nid, np + 2, gamma, leaf_child_count);
    } else return np;
}

void Tree::prune_self(float_type gamma) {
    vector<int> leaf_child_count(nodes.size(), 0);
    Tree::TreeNode *nodes_data = nodes.host_data();
    int n_pruned = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes_data[i].is_leaf && nodes_data[i].is_valid) {
            n_pruned = try_prune_leaf(i, n_pruned, gamma, leaf_child_count);
        }
    }
    LOG(DEBUG) << string_format("%d nodes are pruned", n_pruned);
    reorder_nid();
}


void DeltaTree::prune_self(float_type gamma) {
    vector<int> leaf_child_count(nodes.size(), 0);
    int n_pruned = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes[i].is_leaf && nodes[i].is_valid) {
            n_pruned = try_prune_leaf(i, n_pruned, gamma, leaf_child_count);
        }
    }
    LOG(DEBUG) << string_format("%d nodes are pruned", n_pruned);
    reorder_nid();
}

void DeltaTree::init_CPU(const SyncArray<GHPair> &gradients, const DeltaBoostParam &param, float_type &gain_coef) {
    TIMED_FUNC(timerObj);
    init_structure(param.depth);
    //init root node
//    GHPair sum_gh = thrust::reduce(thrust::host, gradients.host_data(), gradients.host_end());
//    float_type sum_g2 = std::reduce(std::execution::par, gradients.host_data(), gradients.host_end(), 0.0,
//                                    [](const GHPair &a, const GHPair &b){
//        return  a.g * a.g + b.g * b.g;
//    });
    for (auto &gh: gradients.to_vec()) {
        assert(gh.h >= 0);
    }

//    float_type sum_g2 = std::accumulate(gradients.host_data(), gradients.host_end(), 0.0,
//                                         [](float_type a, const GHPair &b){ return  a + b.g * b.g;});
    GHPair sum_gh = std::accumulate(gradients.host_data(), gradients.host_end(), GHPair());
    GHPair sum_rest_gh = std::accumulate(gradients.host_data() + 11, gradients.host_end(), GHPair());

    float_type lambda = param.lambda;
    DeltaNode &root_node = nodes[0];
    root_node.sum_gh_pair = sum_gh;
//    root_node.sum_g2 = sum_g2;
    root_node.is_valid = true;
    root_node.calc_weight_(lambda, g_bin_width, h_bin_width);
    root_node.n_instances = static_cast<int>(gradients.size());
    root_node.potential_nodes_indices.emplace_back(0);

//    gain_coef = sum_g2 / (sum_gh.h + lambda);
}

void DeltaTree::init_structure(int depth) {
    DeltaNode root_node;
    root_node.is_leaf = false;
    root_node.lch_index = -1;
    root_node.rch_index = -1;
    root_node.parent_index = -1;
    root_node.split_feature_id = -1;
    root_node.final_id = 0;
    root_node.is_valid = false;
    root_node.n_instances = 0;
    nodes = {root_node};

//    int n_max_nodes = static_cast<int>(pow(2, depth + 1) - 1);
//    nodes = vector<DeltaNode>(n_max_nodes);
//#pragma omp parallel for
//    for (int i = 0; i < n_max_nodes; i++) {
//        nodes[i].final_id = i;
//        nodes[i].split_feature_id = -1;
//        nodes[i].is_valid = false;
//        nodes[i].parent_index = i == 0 ? -1 : (i - 1) / 2;
//        nodes[i].n_instances = 0;
//        if (i < n_max_nodes / 2) {
//            nodes[i].is_leaf = false;
//            nodes[i].lch_index = i * 2 + 1;
//            nodes[i].rch_index = i * 2 + 2;
//        } else {
//            //leaf nodes
//            nodes[i].is_leaf = true;
//            nodes[i].lch_index = -1;
//            nodes[i].rch_index = -1;
//        }
//    }
}


int DeltaTree::try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count) {
    int p_nid = nodes[nid].parent_index;
    if (p_nid == -1) return np;// is root
    DeltaNode &p_node = nodes[p_nid];
    DeltaNode &lch = nodes[p_node.lch_index];
    DeltaNode &rch = nodes[p_node.rch_index];
    leaf_child_count[p_nid]++;
    if (leaf_child_count[p_nid] >= 2 && p_node.gain.gain_value < gamma) {
        //do pruning
        //delete two children
        CHECK(lch.is_leaf);
        CHECK(rch.is_leaf);
        lch.is_pruned = true;
        rch.is_pruned = true;
        //make parent to leaf
        p_node.is_leaf = true;
        return try_prune_leaf(p_nid, np + 2, gamma, leaf_child_count);
    } else return np;
}

void DeltaTree::reorder_nid() {
    int nid = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        if (nodes[i].is_valid && !nodes[i].is_pruned) {
            nodes[i].final_id = nid;
            nid++;
        }
    }
}