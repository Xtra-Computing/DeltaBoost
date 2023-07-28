from typing import List
import abc

import numpy as np
from joblib import Parallel, delayed


class Evaluative(abc.ABC):
    @abc.abstractmethod
    def predict_score(self, X):
        pass


class Node:
    def __init__(self, is_leaf=False, split_feature_id=None, split_value=None, base_weight=None, default_right=False,
                 lch_id=-1, rch_id=-1, parent_id=-1, n_instances=0, split_nbr=None, split_bid=-1, final_id=-1):
        self.parent_id = parent_id
        self.lch_id = lch_id
        self.rch_id = rch_id
        self.split_feature_id = split_feature_id
        self.split_value = split_value
        self.base_weight = base_weight
        self.default_right = default_right
        self.is_leaf = is_leaf
        self.n_instances = n_instances
        self.split_nbr = split_nbr
        self.split_bid = split_bid
        self.final_id = final_id

    def decide_right(self, x) -> bool:
        """
        :param x: data for prediction
        :return: false for left, true for right
        """
        feature_value = x[self.split_feature_id]
        if abs(feature_value) < 1e-10:
            # missing value
            return self.default_right
        return not (feature_value < self.split_value)

    @classmethod
    def load_from_json(cls, js: dict):
        if js['is_leaf']:
            min_split_bid = max_split_bid = 0
        else:
            min_split_bid = js['split_nbr']['split_bids'][0]
            max_split_bid = js['split_nbr']['split_bids'][-1]
        return cls(parent_id=int(js['parent_index']),
                   lch_id=int(js['lch_index']),
                   rch_id=int(js['rch_index']),
                   split_feature_id=int(js['split_feature_id']),
                   split_value=float(js['split_value']),
                   base_weight=float(js['base_weight']),
                   default_right=bool(js['default_right']),
                   is_leaf=bool(js['is_leaf']),
                   n_instances=js['n_instances'],
                   split_nbr=(min_split_bid, max_split_bid + 1),
                   split_bid=js['split_bid'],
                   final_id=js['final_id'])

    def __repr__(self):
        return f"{self.parent_id=} {self.lch_id=} {self.rch_id=} {self.split_feature_id=} {self.split_value=} " \
               f"{self.base_weight=} {self.default_right=} {self.is_leaf=} {self.n_instances=}"

    def __eq__(self, other):
        if self.is_leaf != other.is_leaf:
            return False

        if self.is_leaf:
            return np.isclose(self.base_weight, other.base_weight, atol=1e-4)
        else:
            return self.split_feature_id == other.split_feature_id \
               and np.isclose(self.split_value, other.split_value, atol=1e-4) \
        # and self.default_right == other.default_right \


class Tree:
    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes if nodes is not None else []

    def add_child_(self, node: Node, is_right):
        self.nodes.append(node)
        if is_right:
            self.nodes[node.parent_id].rch_id = len(self.nodes) - 1
        else:
            self.nodes[node.parent_id].lch_id = len(self.nodes) - 1

    def add_root_(self, node: Node):
        assert self.nodes is None or len(self.nodes) == 0
        self.nodes = [node]

    def predict(self, x):
        node = self.nodes[0]
        while not node.is_leaf:
            if node.decide_right(x):
                node = self.nodes[node.rch_id]
            else:
                node = self.nodes[node.lch_id]
        return node.base_weight

    @classmethod
    def load_from_json(cls, js: dict):
        tree = cls()
        visiting_node_indices = [0]
        while len(visiting_node_indices) > 0:
            node_id = visiting_node_indices.pop(0)
            node = Node.load_from_json(js['nodes'][node_id])
            is_right_child = int(tree.nodes[node.parent_id].rch_id) == node_id if node.parent_id != -1 else False

            if not node.is_leaf:
                js['nodes'][node.lch_id]['parent_index'] = len(tree.nodes)
                js['nodes'][node.rch_id]['parent_index'] = len(tree.nodes)
                visiting_node_indices += [node.lch_id, node.rch_id]
            if node.parent_id == -1:
                tree.add_root_(node)
            else:
                tree.add_child_(node, is_right_child)
        return tree

    def __eq__(self, other):
        visiting_node_indices = [0]
        other_visiting_node_indices = [0]
        while len(visiting_node_indices) > 0:
            node_id = visiting_node_indices.pop(0)
            other_node_id = other_visiting_node_indices.pop(0)
            node = self.nodes[node_id]
            other_node = other.nodes[other_node_id]
            if node != other_node:
                return False

            assert node.is_leaf == other_node.is_leaf
            if not node.is_leaf:
                visiting_node_indices += [node.lch_id, node.rch_id]
                other_visiting_node_indices += [other_node.lch_id, other_node.rch_id]
        return True


class GBDT(Evaluative):
    def __init__(self, lr=1., trees=None):
        self.lr = lr
        self.trees = trees

    def predict_score(self, X: np.ndarray, n_used_trees=None, n_jobs=1):
        """
        :param n_used_trees: number of used trees
        :param X: 2D array
        :param n_jobs: number of jobs for parallel computing
        :return: y: 1D array
        """
        if n_used_trees is None:
            n_used_trees = len(self.trees)

        # scores = np.zeros(X.shape[0])
        def predict_single(x):
            score = 0
            for tree in self.trees[:n_used_trees]:
                score += tree.predict(x) * self.lr
            return score

        scores = Parallel(n_jobs=n_jobs)(delayed(predict_single)(x) for x in X)

        # for i, x in enumerate(X):
        #     score = 0
        #     for tree in self.trees[:n_used_trees]:
        #         score += tree.predict(x) * self.lr
        #     scores[i] = score
        return scores

    def predict(self, X: np.ndarray, task='bin-cls'):
        if task == 'bin-cls':
            return np.where(self.predict_score(X) > 0.5, 1, 0)
        else:
            assert False, "Task not supported."

    @classmethod
    def load_from_json(cls, js, type='deltaboost'):
        assert type in ['deltaboost', 'gbdt'], "Unsupported type"

        gbdt = cls(lr=float(js['learning_rate']), trees=[])
        for tree_js in js[type]['trees']:
            gbdt.trees.append(Tree.load_from_json(tree_js[0]))
        return gbdt

    def __eq__(self, other):
        for i, (tree, other_tree) in enumerate(zip(self.trees, other.trees)):
            if tree != other_tree:
                return False
        return True
