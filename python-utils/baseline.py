import numpy as np
import pickle
import time
import os
import argparse

import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from train_test_split import load_data



class Record(object):

    def __init__(self, raw_data):
        self.raw_data = raw_data

    @classmethod
    def load_from_file(cls, dataset, version):
        """
        :param dataset:
        :param version: [1%, 0.1%]
        :return:
        """
        with open(f'../share/DART_data/{dataset}/{version}/data', 'rb') as f:
            record = cls(pickle.load(f))
            return record

    '''
    return a list of length 10, which is training seconds for seed from 0 to 9
    '''

    def train_times(self):
        return self.raw_data['training_seconds']

    '''
    return a list of length 10, which is forgetting seconds for seed from 0 to 9
    '''

    def forget_times(self):
        return self.raw_data['forgetting_seconds']

    '''
    return a list of length 10, which is retraining seconds for seed from 0 to 9
    '''

    def retrain_times(self):
        return self.raw_data['retraining_seconds']

    '''
    read real labels for input datasets
    dataset_type: from ['test', 'forget', 'retrain']
    '''

    def get_real_labels(self, dataset_type):
        return self.raw_data[f'{dataset_type}_data_df'][['real']]

    '''
    read data as dataframe to calculate the matrix
    model_type: from ['origin', 'forget', 'retrain']
    dataset_type: from ['test', 'forget', 'retrain']
    '''

    def read(self, model_type, dataset_type):
        return self.raw_data[f'{dataset_type}_data_df'].filter(regex=(f'{model_type}.*'))

def test_sklearn_cls(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-02'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    model = GradientBoostingClassifier(n_estimators=n_trees, max_depth=7)
    model.fit(X, y)
    et = time.time()
    print(f'sklearn GBDT training time: {et - st:.3f}s')
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, np.round(pred))
    print(f'sklearn GBDT error: {1 - acc:.4f}')
    return model


def test_sklearn_reg(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-02'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    model = GradientBoostingRegressor(n_estimators=n_trees, max_depth=7)
    model.fit(X, y)
    et = time.time()
    print(f'sklearn GBDT training time: {et - st:.3f}s')
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'sklearn GBDT error: {1 - rmse:.4f}')
    return model


def test_xgb_cls(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    dtrain = xgb.DMatrix(X, label=y, missing=np.NaN, )
    bst = xgb.train({'tree_method': 'approx', 'objective': 'binary:logistic', 'max_bin': 1000,
                     'eta': 1, 'max_depth': 7}, dtrain,
                    num_boost_round=n_trees)
    et = time.time()
    print(f'XGBoost training time: {et - st:.3f}s')
    pred = bst.predict(xgb.DMatrix(X_test))
    acc = accuracy_score(y_test, np.round(pred))
    print(f'XGBoost error: {1 - acc:.4f}')
    return bst


def test_xgb_reg(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    dtrain = xgb.DMatrix(X, label=y, missing=np.NaN)
    bst = xgb.train({'tree_method': 'approx', 'objective': 'reg:squarederror', 'max_bin': 1000,
                     'eta': 1, 'max_depth': 7}, dtrain,
                    num_boost_round=n_trees)
    et = time.time()
    print(f'XGBoost training time: {et - st:.3f}s')
    pred = bst.predict(xgb.DMatrix(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'XGBoost RMSE: {rmse:.4f}')
    return bst



def test_tree_cls(dataset):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    dt = DecisionTreeClassifier(max_depth=7)
    dt.fit(X, y)
    et = time.time()
    print(f'Decision Tree training time: {et - st:.3f}s')
    pred = dt.predict(X_test)
    acc = accuracy_score(y_test, np.round(pred))
    print(f'Decision Tree error: {1 - acc:.4f}')
    return dt


def test_tree_reg(dataset):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    dt = DecisionTreeRegressor(max_depth=7)
    dt.fit(X, y)
    et = time.time()
    print(f'Decision Tree training time: {et - st:.3f}s')
    pred = dt.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'Decision Tree accuracy: {rmse:.4f}')
    return dt


def test_rf_cls(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=7)
    rf.fit(X, y)
    et = time.time()
    print(f'Random Forest training time: {et - st:.3f}s')
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, np.round(pred))
    print(f'Random Forest error: {1 - acc:.4f}')
    return rf


def test_rf_reg(dataset, n_trees=10):
    train_dataset_path = f'../data/{dataset}.train.remain_1e-03'
    test_dataset_path = f'../data/{dataset}.test'
    X, y = load_data(train_dataset_path, 'csv', scale_y=True, output_dense=True)
    X_test, y_test = load_data(test_dataset_path, 'csv', scale_y=True, output_dense=True)
    st = time.time()
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=7)
    rf.fit(X, y)
    et = time.time()
    print(f'Random Forest training time: {et - st:.3f}s')
    pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'Random Forest accuracy: {rmse:.4f}')
    return rf


if __name__ == '__main__':
    # add arguments for number of trees
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--n_trees', type=int, default=10)
    args = parser.parse_args()

    test_sklearn_cls('codrna', args.n_trees)
    print("=====================================")
    test_sklearn_cls('covtype', args.n_trees)
    print("=====================================")
    test_sklearn_cls('gisette', args.n_trees)
    print("=====================================")
    test_sklearn_reg('cadata', args.n_trees)
    print("=====================================")
    test_sklearn_reg('msd', args.n_trees)
    print("=====================================")

    test_xgb_cls('codrna', args.n_trees)
    print("=====================================")
    test_xgb_cls('covtype', args.n_trees)
    print("=====================================")
    test_xgb_cls('gisette', args.n_trees)
    print("=====================================")
    test_xgb_reg('cadata', args.n_trees)
    print("=====================================")
    test_xgb_reg('msd', args.n_trees)
    print("=====================================")

    test_rf_cls('codrna', args.n_trees)
    print("=====================================")
    test_rf_cls('covtype', args.n_trees)
    print("=====================================")
    test_rf_cls('gisette', args.n_trees)
    print("=====================================")
    test_rf_reg('cadata', args.n_trees)
    print("=====================================")
    test_rf_reg('msd', args.n_trees)

    test_tree_cls('codrna')
    print("=====================================")
    test_tree_cls('covtype')
    print("=====================================")
    test_tree_cls('gisette')
    print("=====================================")
    test_tree_reg('cadata')
    print("=====================================")
    test_tree_reg('msd')

