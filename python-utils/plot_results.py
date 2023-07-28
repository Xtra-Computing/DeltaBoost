import os.path
import re
import pickle
import logging
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ujson as json
from joblib import Parallel, delayed
from tqdm import tqdm

from GBDT import GBDT
from train_test_split import load_data


class Record(object):
    hedgecut_path = "/data/junhui/HedgeCut"
    DART_path = "/data/junhui/DART"
    hedgecut_labels_path = "/data/junhui/HedgeCut/labels"

    def __init__(self, raw_data, model_type, slice_num, dataset):
        self.raw_data = raw_data
        self.model_type = model_type
        self.slice_num = slice_num
        self.dataset = dataset

    @classmethod
    def load_from_file(cls, dataset, version, model_type, slice_num=None):
        # global DART_path
        # global hedgecut_path
        if model_type == "DART":
            path = cls.DART_path
        elif model_type == "hedgecut":
            path = cls.hedgecut_path
        else:
            raise ValueError("model_type must be DART or hedgecut")

        if model_type == "DART":
            with open(path + f'/{dataset}/{version}/data', 'rb') as f:
                record = cls(pickle.load(f), model_type, slice_num, dataset)
                logging.debug("Done loading DART")
                return record

        if model_type == "hedgecut":
            if dataset == 'susy':
                slice_num_ = 100 if slice_num is None else slice_num
                ret = {
                    'vs_origin_test': [],
                    'vs_origin_forget': [],
                    'vs_origin_retrain': [],
                    'vs_retrain_test': [],
                    'vs_retrain_forget': [],
                    'vs_retrain_retrain': [],
                    'vs_forget_test': [],
                    'vs_forget_forget': [],
                    'vs_forget_retrain': []
                }
                for i in range(slice_num_):
                    with open(path + f'/{dataset}/{version}/{i}.json') as f:
                        j = json.load(f)
                        j = json.loads(j)
                        for k, v in j.items():
                            ret[k].append(v[0])
                        record = cls(ret, model_type, slice_num, dataset)
                        logging.debug("Done loading hedgecut")
                return record
            else:
                with open(path + f'/{dataset}/{version}/data.json') as f:
                    j = json.load(f)
                    j = json.loads(j)
                    if slice_num is not None:
                        for k, v in j.items():
                            j[k] = v[:slice_num]
                    record = cls(j, model_type, slice_num, dataset)
                    logging.debug("Done loading hedgecut")
                    return record

    def get_real_labels(self, dataset_type):
        if self.model_type == "hedgecut":
            return pd.read_csv(f'{self.hedgecut_labels_path}/{self.dataset}/test.csv', sep='\t')['label']
        if self.model_type == "DART":
            return self.raw_data[f'{dataset_type}_data_df'][['real']]

    '''
    read data as dataframe to calculate the matrix
    model_type: from ['origin', 'forget', 'retrain']
    dataset_type: from ['test', 'forget', 'retrain']
    '''

    def read(self, model_type, dataset_type):
        if self.model_type == "DART":
            return self.raw_data[f'{dataset_type}_data_df'].filter(regex=(f'{model_type}.*'))
        if self.model_type == "hedgecut":
            return np.array(self.raw_data[f'vs_{model_type}_{dataset_type}'])

    def load_2d_array(self, model_type, dataset_type):
        logging.debug(f"load_2d_array {model_type} {dataset_type}")
        if self.model_type == "DART":
            df = self.read(model_type, dataset_type)
            df_slice = df.to_numpy()[:, :self.slice_num].T
            return df_slice
        if self.model_type == "hedgecut":
            return self.read(model_type, dataset_type)

    def get_accuracy(self, models, dataset):
        if self.model_type == "hedgecut":
            labels = self.get_real_labels(dataset)
            ret = []
            for model in models:
                arr = self.raw_data[f'vs_{model}_{dataset}']
                df = pd.concat([pd.Series(i) for i in arr], axis=1).applymap(lambda x: 1.0 if x > 0.5 else 0.0)
                acc = (df.sub(labels, axis=0).abs().sum() / (len(df))).values
                ret.append((np.mean(acc), np.std(acc)))
            return ret
        if self.model_type == "DART":
            def get_real_labels(self, dataset):
                return self.raw_data[f'{dataset}_data_df'][['real']]

            labels = self.get_real_labels(dataset)['real']
            ret = []
            for model in models:
                df = self.read(model, dataset)
                x = df.applymap(lambda x: 0.0 if x >= 0.5 else 1.0)
                acc = (x.sub(labels, axis=0).abs().sum() / (len(df))).values
                ret.append((np.mean(acc), np.std(acc)))
            return ret


def get_scores_from_file(file_path, out_fmt='str') -> tuple[str, list]:
    with open(file_path, 'r') as f:
        f_text = f.read()
        metric = re.match(".*INFO (?:deltaboost|gbdt)\.cpp:[0-9]* : Test: (.*) =", f_text)
        pattern = re.compile(
            ".*INFO (?:deltaboost|gbdt)\.cpp:[0-9]* :.*(?:error|RMSE) = ([+-]?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)")
        scores = pattern.findall(f_text)

        if out_fmt == 'float':
            scores = [float(s) for s in scores]
        return metric, scores


def plot_score_before_after_removal(out_dir, datasets, remove_ratios, save_path=None, plot_type='table'):
    assert plot_type in ['raw', 'fig', 'table']
    summary = []
    for dataset in datasets:
        df_ratios = []
        for raw_ratio in remove_ratios:
            ratio = "0.01" if raw_ratio == '1e-02' else "0.001"
            out_path = os.path.join(out_dir, f"{dataset}_deltaboost_{ratio}.out")
            out_retrain_path = os.path.join(out_dir, f"{dataset}_deltaboost_{ratio}_retrain.out")
            metric, out_data = get_scores_from_file(out_path, out_fmt='float')
            assert len(out_data) == 6, f"{out_path}, got {len(out_data)}"
            scores = [float(x) for x in out_data[:3]]
            scores_del = [float(x) for x in out_data[3:]]
            _, out_retrain_data = get_scores_from_file(out_retrain_path, out_fmt='float')
            scores_retrain = out_retrain_data[:3]

            # reorder the result
            scores = np.array([scores[2], scores[1], scores[0]])
            scores_del = np.array([scores_del[2], scores_del[1], scores_del[0]])
            scores_retrain = np.array([scores_retrain[2], scores_retrain[1], scores_retrain[0]])

            x_labels = [r'$D_r$', r'$D_f$', r'$D_{test}$']
            df = pd.DataFrame(data={'Original': scores, 'Remove (ours)': scores_del,
                                    'Retrain (target)': scores_retrain})
            if plot_type == 'fig':
                df.set_index(x_labels)
                ax = df.plot(kind='bar', rot=0, xlabel="Datset partition", ylabel=metric,
                             title=f"{dataset} removing {ratio}")
                ax.legend()
                ax.margins(y=0.3)
                plt.show()
            elif plot_type in ['table']:
                columns = df.columns
                df = df.T
                df['Dataset'] = [f"\\multirow{{3}}{{*}}{{{dataset}}}", "", ""]
                df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
                df.set_index('Dataset', inplace=True)
                df_ratios.append(df)
        df_combine = pd.concat(df_ratios, axis=1)
        df_combine.insert(0, 'Method')

        df_combine.style.format(lambda x: f"{x:.6f}")
        # df_combine = df_combine[['Method', r'$D$', r'$D_f$', r'$D_r$', r'$D$', r'$D_f$', r'$D_r$']]
        summary.append('\n'.join(df_combine.style.to_latex().split('\n')[3:6]))
    if plot_type == 'table':
        latex_table = '\n\\midrule\n'.join(summary)
        print(latex_table)


def plot_deltaboost_vs_gbdt(out_dir, datasets, save_path=None, present='test', n_trees=10, n_rounds=1):
    assert present in ['test']

    if n_trees == 10:
        # 10 trees
        xgboost_scores = {
            'codrna': 1 - 0.9552,
            'covtype': 1 - 0.8016,
            'gisette': 1 - 0.9630,
            'cadata': 0.1165,
            'msd': 0.1143,
        }
        rf_scores = {
            'codrna': 0.1011,
            'covtype': 0.2472,
            'gisette': 0.0590,
            'cadata': 0.1307,
            'msd': 0.1170,
        }
        dt_scores = {
            'codrna': 0.0670,
            'covtype': 0.2225,
            'gisette': 0.0750,
            'cadata': 0.1382,
            'msd': 0.1185,
        }
    elif n_trees == 100:
        xgboost_scores = {
            'codrna': 0.0343,
            'covtype': 0.0609,
            'gisette': 0.0290,
            'cadata': 0.1272,
            'msd': 0.1219,
        }
        rf_scores = {
            'codrna': 0.1287,
            'covtype': 0.2402,
            'gisette': 0.0490,
            'cadata': 0.1283,
            'msd': 0.1169,
        }
        dt_scores = {
            'codrna': 0.0670,
            'covtype': 0.2225,
            'gisette': 0.0750,
            'cadata': 0.1382,
            'msd': 0.1185,
        }
    else:
        assert False

    summary = []
    for dataset in datasets:
        ratio = '1e-03'  # either should be the same
        deltaboost_scores = []
        for i in range(n_rounds):
            out_deltaboost_path = os.path.join(out_dir, f"tree{n_trees}/{dataset}_deltaboost_{ratio}_{i}.out")
            metric, deltaboost_data = get_scores_from_file(out_deltaboost_path, out_fmt='float')
            print(metric, deltaboost_data)
            deltaboost_scores.append(deltaboost_data[0])
        db_mean = np.mean(deltaboost_scores)
        out_gbdt_path = os.path.join(out_dir, f"tree{n_trees}/{dataset}_gbdt_{ratio}.out")
        _, gbdt_data = get_scores_from_file(out_gbdt_path, out_fmt='float')
        print(gbdt_data)
        gbdt_score = gbdt_data[0]
        summary.append([db_mean, gbdt_score, xgboost_scores[dataset], rf_scores[dataset], dt_scores[dataset]])

        print(f"{dataset} done.")

        # # empty datasets
        # if len(deltaboost_scores) == 0 or len(gbdt_scores) == 0:
        #     if present == 'test':
        #         summary.append([0, 0])
        #     else:
        #         assert False
        # else:
        #     if present == 'test':
        #         summary.append([gbdt_scores[0], deltaboost_scores[0]])
        #     else:
        #         assert False
    # plot
    if present == 'test':
        algos = ['DeltaBoost', 'ThunderGBM-CPU', 'XGBoost', 'Random Forest', 'Decision Tree']
        summary_df = pd.DataFrame(data=summary, index=datasets, columns=algos)
        ax = summary_df.plot(kind='bar', rot=0, xlabel='Datasets', ylabel='Error',
                             title=f'Error of DeltaBoost and GBDT (tree {n_trees})',
                             figsize=None)
        ax.margins(y=0.1)
        ax.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()



def summary_model_diff_for_dataset(dataset, n_trees, remove_ratios):
    """
    Evaluate the model difference by inference. This function can be very slow.
    :param dataset:
    :param n_trees:
    :param remove_ratios:
    :return:
    """
    df_ratios = []
    for remove_ratio in remove_ratios:
        print(f"Remove ratio {remove_ratio}:")
        print("Loading models.")
        model_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deltaboost.json"
        model_remain_path = f"../cache/{dataset}_tree{n_trees}_retrain_{remove_ratio}_deltaboost.json"
        model_deleted_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deleted.json"
        with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2, open(model_deleted_path) as f3:
            js1 = json.load(f1)
            js2 = json.load(f2)
            js3 = json.load(f3)
        gbdt_original = GBDT.load_from_json(js1, 'deltaboost')
        gbdt_retrain = GBDT.load_from_json(js2, 'deltaboost')
        gbdt_deleted = GBDT.load_from_json(js3, 'deltaboost')

        print("Loading dataset.")
        # train_X, train_y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', output_dense=True)
        remain_X, remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='csv',
                                       output_dense=True)
        delete_X, delete_y = load_data(f"../data/{dataset}.train.delete_{remove_ratio}", data_fmt='csv',
                                       output_dense=True)
        test_X, test_y = load_data(f"../data/{dataset}.test", data_fmt='csv', output_dense=True)

        print("Prediction on remain")
        original_score_on_remain = gbdt_original.predict_score(remain_X)
        retrain_score_on_remain = gbdt_retrain.predict_score(remain_X)
        deleted_score_on_remain = gbdt_deleted.predict_score(remain_X)
        diff_retrain_original_on_remain = np.average(np.abs(original_score_on_remain - retrain_score_on_remain))
        diff_retrain_delete_on_remain = np.average(np.abs(retrain_score_on_remain - deleted_score_on_remain))
        print("Prediction on delete")
        original_score_on_delete = gbdt_original.predict_score(delete_X)
        retrain_score_on_delete = gbdt_retrain.predict_score(delete_X)
        deleted_score_on_delete = gbdt_deleted.predict_score(delete_X)
        diff_retrain_original_on_delete = np.average(np.abs(original_score_on_delete - retrain_score_on_delete))
        diff_retrain_delete_on_delete = np.average(np.abs(retrain_score_on_delete - deleted_score_on_delete))
        print("Prediction on test")
        original_score_on_test = gbdt_original.predict_score(test_X)
        retrain_score_on_test = gbdt_retrain.predict_score(test_X)
        deleted_score_on_test = gbdt_deleted.predict_score(test_X)
        diff_retrain_original_on_test = np.average(np.abs(original_score_on_test - retrain_score_on_test))
        diff_retrain_delete_on_test = np.average(np.abs(retrain_score_on_test - deleted_score_on_test))

        x_labels = [r'$D_f$', r'$D_r$', r'$D_{test}$']
        df = pd.DataFrame(data={
            'original vs. retrain': [diff_retrain_original_on_remain, diff_retrain_original_on_delete,
                                     diff_retrain_original_on_test],
            'delete vs. retrain': [diff_retrain_delete_on_remain, diff_retrain_delete_on_delete,
                                   diff_retrain_delete_on_test]})
        columns = df.columns
        df = df.T
        df['Dataset'] = [f"\\multirow{{2}}{{*}}{{{dataset}}}", ""]
        df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
        df.set_index('Dataset', inplace=True)
        df_ratios.append(df)
    df_combine = pd.concat(df_ratios, axis=1)
    df_combine.insert(0, 'Method', columns)
    return df_combine


def print_model_diff(datasets, n_trees, remove_ratios):
    summary = []
    for dataset in datasets:
        df_combine = summary_model_diff_for_dataset(dataset, n_trees, remove_ratios)
        df_combine.style.format(lambda x: f"{x:.6f}")
        summary.append('\n'.join(df_combine.style.to_latex().split('\n')[3:5]))
    latex_table = '\n\\midrule\n'.join(summary)
    print(latex_table)


class ModelDiffSingle:
    def __init__(self, dataset, n_trees, remove_ratio, n_rounds, keyword, record: Record = None, n_jobs=1,
                 deltaboost_path="../cache", deltaboost_predict=False):
        """
        :param dataset: dataset name
        :param n_trees: number of trees
        :param remove_ratio: remove ratio, 1e-02 or 1e-03
        :param n_rounds: number of random rounds
        :param keyword: ['original', 'retrain', 'delete']
        :param record: None means load model and inference on the data, otherwise load results from record
        :param n_jobs: number of jobs in parallel
        """
        self.dataset = dataset
        self.n_trees = n_trees
        self.remove_ratio = remove_ratio
        self.n_rounds = n_rounds
        self.keyword = keyword
        self.record = record
        self.n_jobs = n_jobs
        self.deltaboost_path = deltaboost_path
        self.deltaboost_predict = deltaboost_predict

        if record is None:
            if deltaboost_predict:
                # load data from prediction
                logging.debug("Loading deltaboost data")
                if keyword in ['remain', 'delete']:
                    self.X, self.y = load_data(f"../data/{dataset}.train.{keyword}_{remove_ratio}",
                                               data_fmt='csv',
                                               output_dense=True)
                elif keyword == 'test':
                    self.X, self.y = load_data(f"../data/{dataset}.test", data_fmt='csv', output_dense=True)
                else:
                    raise ValueError(f"Invalid keyword: {keyword}")
                self.original_score = np.zeros((n_rounds, self.X.shape[0]))
                self.retrain_score = np.zeros((n_rounds, self.X.shape[0]))
                self.deleted_score = np.zeros((n_rounds, self.X.shape[0]))
                logging.debug("Done loading deltaboost data")
            else:
                # get size from example
                example_output = np.genfromtxt(
                    f"{deltaboost_path}/{dataset}_tree{n_trees}_original_{self.remove_ratio}_0_deltaboost_score_{keyword}.csv",
                    delimiter=',')
                n_instance = example_output.shape[0]

                # directly load output of deltaboost from csv
                logging.debug("Loading deltaboost output.")
                self.original_score = np.zeros((n_rounds, n_instance))
                self.retrain_score = np.zeros((n_rounds, n_instance))
                self.deleted_score = np.zeros((n_rounds, n_instance))
                self.X, self.y = None, None
                for i in tqdm(range(n_rounds)):
                    self.original_score[i, :] = np.genfromtxt(
                        f"{deltaboost_path}/{dataset}_tree{n_trees}_original_{self.remove_ratio}_{i}_deltaboost_score_{keyword}.csv",
                        delimiter=',')[:, 0]
                    self.retrain_score[i, :] = np.genfromtxt(
                        f"{deltaboost_path}/{dataset}_tree{n_trees}_retrain_{self.remove_ratio}_{i}_deltaboost_score_{keyword}.csv",
                        delimiter=',')[:, 0]
                    self.deleted_score[i, :] = np.genfromtxt(
                        f"{deltaboost_path}/{dataset}_tree{n_trees}_original_{self.remove_ratio}_{i}_deleted_score_{keyword}.csv",
                        delimiter=',')[:, 0]
                logging.debug("Done loading deltaboost output.")
        else:
            # read from Record
            logging.debug("Loading from record")
            self.X, self.y = None, None
            self.original_score = record.load_2d_array('origin', 'test')
            self.retrain_score = record.load_2d_array('retrain', 'test')
            self.deleted_score = record.load_2d_array('forget', 'test')
            logging.debug("Done loading from record")

    def predict_i(self, i, n_used_trees=None):
        if self.X is None or self.y is None:
            raise ValueError("X and y should not be None")

        model_original_path = os.path.join(self.deltaboost_path,
                                           f"{self.dataset}_tree{self.n_trees}_original_{self.remove_ratio}_{i}_deltaboost.json")
        model_retrain_path = os.path.join(self.deltaboost_path,
                                          f"{self.dataset}_tree{self.n_trees}_retrain_{self.remove_ratio}_{i}_deltaboost.json")
        model_deleted_path = os.path.join(self.deltaboost_path,
                                          f"{self.dataset}_tree{self.n_trees}_original_{self.remove_ratio}_{i}_deleted.json")

        # load model
        with open(model_original_path, 'r') as f1, open(model_retrain_path, 'r') as f2, open(model_deleted_path,
                                                                                             'r') as f3:
            js1 = json.load(f1)
            js2 = json.load(f2)
            js3 = json.load(f3)
        gbdt_original = GBDT.load_from_json(js1)
        gbdt_retrain = GBDT.load_from_json(js2)
        gbdt_deleted = GBDT.load_from_json(js3)

        # predict
        original_score = gbdt_original.predict_score(self.X, n_used_trees, self.n_jobs)
        retrain_score = gbdt_retrain.predict_score(self.X, n_used_trees, self.n_jobs)
        deleted_score = gbdt_deleted.predict_score(self.X, n_used_trees, self.n_jobs)
        return original_score, retrain_score, deleted_score

    def predict_(self, n_used_trees=None):
        logging.debug("predicting with deltaboost")
        # execute predict_i in parallel with joblib.Parallel
        results = Parallel(n_jobs=1)(delayed(self.predict_i)(i, n_used_trees) for i in range(self.n_rounds))

        # convert results to numpy array
        self.original_score, self.retrain_score, self.deleted_score = np.array(results).transpose((1, 0, 2))
        logging.debug("Done")

    def get_hellinger_distance(self, n_bins=50, return_std=True):
        logging.debug("Calculating Hellinger distance")
        min_value = min(np.min(self.original_score), np.min(self.retrain_score), np.min(self.deleted_score))
        max_value = max(np.max(self.original_score), np.max(self.retrain_score), np.max(self.deleted_score))

        n_rounds = self.original_score.shape[0]
        n_instances = self.original_score.shape[1]

        if n_rounds > 1000:
            warnings.warn("n_rounds is larger than 1000, it may take a long time to calculate Hellinger distance")

        def get_hist_i(i):
            """
            :param i: instance index
            :return: normalized histogram counts
            """
            # if i == 81:
            #     return np.nan, np.nan, np.nan
            original_hist_i = np.histogram(self.original_score[:, i], bins=n_bins,
                                           range=(min_value, max_value), density=True)[0]
            retrain_hist_i = np.histogram(self.retrain_score[:, i], bins=n_bins,
                                          range=(min_value, max_value), density=True)[0]
            deleted_hist_i = np.histogram(self.deleted_score[:, i], bins=n_bins,
                                          range=(min_value, max_value), density=True)[0]
            return original_hist_i, retrain_hist_i, deleted_hist_i

        # Get the hist of all instances in parallel by joblib.Parallel
        results = Parallel(n_jobs=self.n_jobs)(delayed(get_hist_i)(i) for i in range(n_instances))
        # convert results to numpy array
        original_hist, retrain_hist, deleted_hist = np.array(results).transpose((1, 0, 2))

        # Hellinger distance
        bin_width = (max_value - min_value) / n_bins
        retrain_vs_deleted = 1 - bin_width * np.sum(np.sqrt(retrain_hist * deleted_hist), axis=1)
        original_vs_retrain = 1 - bin_width * np.sum(np.sqrt(original_hist * retrain_hist), axis=1)

        logging.debug("Done")

        if return_std:
            return (np.nanmean(original_vs_retrain), np.nanstd(original_vs_retrain)), \
                   (np.nanmean(retrain_vs_deleted), np.nanstd(retrain_vs_deleted))
        else:
            return np.nanmean(original_vs_retrain), np.nanmean(retrain_vs_deleted)

    def get_accuracy(self):
        logging.debug("Loading Accuracy")
        ret = self.record.get_accuracy(['origin', 'forget', 'retrain'], 'test')
        logging.debug("Done")
        return ret


class ModelDiff:
    def __init__(self, datasets, remove_ratios, n_trees, n_rounds, n_used_trees=None, keyword='test', n_jobs=1,
                 hedgecut_path=None, dart_path=None, deltaboost_path="../cache/", deltaboost_out_path=None,
                 deltaboost_predict=False, forget_table_cache_path=None, accuracy_table_cache_path=None):
        """
        Manage model diff of three methods: DeltaBoost, HedgeCut, DaRE
        :param datasets: list of dataset names, e.g. cadata
        :param remove_ratios: list of remove ratios, e.g. 1e-02
        :param n_trees: number of trees
        :param n_rounds: number of rounds
        :param n_used_trees: number of used trees, if None, use all trees
        :param n_jobs: number of jobs for parallel computing
        :param hedgecut_path: path of hedgecut model, if None, use default path
        :param dart_path: path of dart model, if None, use default path
        :param deltaboost_path: path of deltaboost model, if None, use default path
        :param deltaboost_predict: whether to predict deltaboost model
        :param table_cache_path: path of table cache, if None, load from scratch
        :param update_hedgecut: whether to update hedgecut data
        :param update_dart: whether to update dart data
        :param update_deltaboost: whether to update deltaboost data
        """
        self.datasets = datasets
        self.remove_ratios = remove_ratios
        self.n_trees = n_trees
        self.n_rounds = n_rounds
        self.n_used_trees = n_used_trees
        self.n_jobs = n_jobs
        self.keyword = keyword
        self.hedgecut_path = hedgecut_path
        self.dart_path = dart_path
        self.deltaboost_path = deltaboost_path
        self.deltaboost_out_path = deltaboost_out_path
        self.deltaboost_predict = deltaboost_predict

        if forget_table_cache_path is None or os.path.isfile(forget_table_cache_path):
            self.forget_table_cache_path = forget_table_cache_path
        else:
            self.forget_table_cache_path = None
            warnings.warn("Forget table cache path not found, calculate from scratch")
        if accuracy_table_cache_path is None or os.path.isfile(accuracy_table_cache_path):
            self.accuracy_table_cache_path = accuracy_table_cache_path
        else:
            self.accuracy_table_cache_path = None
            warnings.warn("Accuracy table cache path not found, calculate from scratch")

        if forget_table_cache_path is None:
            self.table_data = np.zeros([len(datasets) * 2, len(remove_ratios) * 3], dtype='U32')
        else:
            self.table_data = np.genfromtxt(forget_table_cache_path, dtype='U32', delimiter=',')

        if accuracy_table_cache_path is None:
            self.accuracy_table = np.zeros([len(datasets) * 3, len(remove_ratios) * 3], dtype='U32')
        else:
            self.accuracy_table = np.genfromtxt(accuracy_table_cache_path, dtype='U32', delimiter=',')

        if hedgecut_path is not None:
            Record.hedgecut_path = hedgecut_path
        if dart_path is not None:
            Record.dart_path = dart_path

    def get_raw_data_forget_(self, n_bins=50, save_path=None, update_hedgecut=None, update_dart=None,
                             update_deltaboost=None,
                             update_datasets=None):
        """
        Get raw data of three methods and stored in self.table_data
        :param n_bins: number of bins for histogram
        :param save_path: path to save forget table
        :param update_hedgecut: whether to update hedgecut model
        :param update_dart: whether to update dart model
        :param update_deltaboost: whether to update deltaboost model
        :param update_datasets: the list of datasets to update
        :return:
        """
        # initialize parameters
        if self.forget_table_cache_path is None:
            _update_dart = _update_hedgecut = _update_deltaboost = True  # default update all
        else:
            _update_dart = _update_hedgecut = _update_deltaboost = False  # default to load from cache
        # overwrite parameters if specified
        _update_dart = _update_dart if update_dart is None else update_dart
        _update_hedgecut = _update_hedgecut if update_hedgecut is None else update_hedgecut
        _update_deltaboost = _update_deltaboost if update_deltaboost is None else update_deltaboost

        ratio2version = {'1e-03': '0.1%', '1e-02': '1%'}
        for i, dataset in enumerate(self.datasets):
            for j, remove_ratio in enumerate(self.remove_ratios):
                if dataset in ['codrna', 'gisette', 'covtype', 'higgs', 'susy']:
                    if _update_dart and (update_datasets is None or dataset in update_datasets):
                        # load dart from record
                        logging.debug(f"Loading dart record")
                        record_dart = Record.load_from_file(dataset, ratio2version[remove_ratio], 'DART', self.n_rounds)
                        logging.debug(f"Done loading, calculating Hellinger distance")
                        model_diff_dart = ModelDiffSingle(dataset, self.n_trees, remove_ratio, self.n_rounds,
                                                          self.keyword, n_jobs=self.n_jobs, record=record_dart)
                        ovr_dart_data, rvd_dart_data = model_diff_dart.get_hellinger_distance(return_std=True,
                                                                                              n_bins=n_bins)
                        ovr_dart: str = f"{ovr_dart_data[0]:.4f}\\textpm {ovr_dart_data[1]:.4f}"
                        rvd_dart: str = f"{rvd_dart_data[0]:.4f}\\textpm {rvd_dart_data[1]:.4f}"
                        logging.info(f"{dataset} {remove_ratio} dart done.")

                    else:
                        logging.debug(f"Loading dart results from cache")
                        ovr_dart = self.table_data[i * 2, j * 3]
                        rvd_dart = self.table_data[i * 2 + 1, j * 3]

                    if _update_hedgecut and (update_datasets is None or dataset in update_datasets):
                        logging.info(f"{dataset} {remove_ratio} starts getting raw data.")
                        # load hedgecut from record
                        logging.debug(f"Loading hedgecut record")
                        record_hedgecut = Record.load_from_file(dataset, ratio2version[remove_ratio], 'hedgecut',
                                                                self.n_rounds)
                        logging.debug(f"Done loading, calculating Hellinger distance")
                        model_diff_hedgecut = ModelDiffSingle(dataset, self.n_trees, remove_ratio, self.n_rounds,
                                                              self.keyword, n_jobs=self.n_jobs, record=record_hedgecut)
                        ovr_hedgecut_data, rvd_hedgecut_data = model_diff_hedgecut.get_hellinger_distance(
                            return_std=True, n_bins=n_bins)
                        ovr_hedgecut: str = f"{ovr_hedgecut_data[0]:.4f}\\textpm {ovr_hedgecut_data[1]:.4f}"
                        rvd_hedgecut: str = f"{rvd_hedgecut_data[0]:.4f}\\textpm {rvd_hedgecut_data[1]:.4f}"
                        logging.info(f"{dataset} {remove_ratio} hedgecut done.")
                    else:
                        logging.debug(f"Loading hedgecut results from cache")
                        ovr_hedgecut = self.table_data[i * 2, j * 3 + 1]
                        rvd_hedgecut = self.table_data[i * 2 + 1, j * 3 + 1]
                else:
                    ovr_hedgecut = rvd_hedgecut = ovr_dart = rvd_dart = '-'
                    logging.info(f"{dataset} {remove_ratio} HedgeCut and DART skipped.")

                if _update_deltaboost and (update_datasets is None or dataset in update_datasets):
                    try:
                        # load deltaboost by inference or outputs
                        model_diff_deltaboost = ModelDiffSingle(dataset, self.n_trees, remove_ratio, self.n_rounds,
                                                                self.keyword, n_jobs=self.n_jobs,
                                                                deltaboost_path=self.deltaboost_path,
                                                                deltaboost_predict=self.deltaboost_predict)
                        if self.deltaboost_predict:
                            model_diff_deltaboost.predict_(self.n_used_trees)
                        ovr_deltaboost_data, rvd_deltaboost_data = model_diff_deltaboost.get_hellinger_distance(
                            return_std=True,
                            n_bins=n_bins)
                        ovr_deltaboost: str = f"{ovr_deltaboost_data[0]:.4f}\\textpm {ovr_deltaboost_data[1]:.4f}"
                        rvd_deltaboost: str = f"{rvd_deltaboost_data[0]:.4f}\\textpm {rvd_deltaboost_data[1]:.4f}"
                    except Exception as e:
                        logging.error(f"{dataset} {remove_ratio} deltaboost failed: {e}")
                        ovr_deltaboost = rvd_deltaboost = '-'
                else:
                    logging.debug(f"Loading deltaboost results from cache")
                    ovr_deltaboost = self.table_data[i * 2, j * 3 + 2]
                    rvd_deltaboost = self.table_data[i * 2 + 1, j * 3 + 2]

                # store data
                self.table_data[2 * i, 3 * j] = ovr_dart
                self.table_data[2 * i + 1, 3 * j] = rvd_dart
                self.table_data[2 * i, 3 * j + 1] = ovr_hedgecut
                self.table_data[2 * i + 1, 3 * j + 1] = rvd_hedgecut
                self.table_data[2 * i, 3 * j + 2] = ovr_deltaboost
                self.table_data[2 * i + 1, 3 * j + 2] = rvd_deltaboost

                logging.info(f"{dataset}, {remove_ratio} done.")

        if save_path is not None:
            np.savetxt(save_path, self.table_data, fmt='%s', delimiter=',')
            logging.info(f"Table saved to {save_path}")

    def get_raw_data_accuracy_(self, save_path=None, update_hedgecut=None, update_dart=None, update_deltaboost=None,
                               update_datasets=None):
        """
        Get raw accuracy data of three methods and stored in self.table_data
        :return:
        """
        # initialize parameters
        if self.accuracy_table_cache_path is None:
            _update_dart = _update_hedgecut = _update_deltaboost = True  # default update all
        else:
            _update_dart = _update_hedgecut = _update_deltaboost = False  # default to load from cache
        # overwrite parameters if specified
        _update_dart = _update_dart if update_dart is None else update_dart
        _update_hedgecut = _update_hedgecut if update_hedgecut is None else update_hedgecut
        _update_deltaboost = _update_deltaboost if update_deltaboost is None else update_deltaboost

        ratio2version = {'1e-03': '0.1%', '1e-02': '1%'}
        for i, dataset in enumerate(self.datasets):
            for j, remove_ratio in enumerate(self.remove_ratios):
                if dataset in ['codrna', 'gisette', 'covtype', 'higgs', 'susy']:
                    if _update_dart and (update_datasets is None or dataset in update_datasets):
                        # load dart from record
                        logging.debug(f"Loading dart record")
                        record_dart = Record.load_from_file(dataset, ratio2version[remove_ratio], 'DART', self.n_rounds)
                        logging.debug(f"Done loading, calculating Hellinger distance")
                        model_diff_dart = ModelDiffSingle(dataset, self.n_trees, remove_ratio, self.n_rounds,
                                                          self.keyword, n_jobs=self.n_jobs, record=record_dart)
                        accs_dart = model_diff_dart.get_accuracy()
                        acc_dart_origin = f"{accs_dart[0][0]:.4f}\\textpm {accs_dart[0][1]:.4f}"
                        acc_dart_forget = f"{accs_dart[1][0]:.4f}\\textpm {accs_dart[1][1]:.4f}"
                        acc_dart_retrain = f"{accs_dart[2][0]:.4f}\\textpm {accs_dart[2][1]:.4f}"
                        logging.info(f"{dataset} {remove_ratio} dart done.")

                    else:
                        logging.debug(f"Loading dart results from cache")
                        acc_dart_origin = self.accuracy_table[i * 3, j * 3]
                        acc_dart_forget = self.accuracy_table[i * 3 + 1, j * 3]
                        acc_dart_retrain = self.accuracy_table[i * 3 + 2, j * 3]

                    if _update_hedgecut and (update_datasets is None or dataset in update_datasets):
                        logging.info(f"{dataset} {remove_ratio} starts getting raw data.")
                        # load hedgecut from record
                        logging.debug(f"Loading hedgecut record")
                        record_hedgecut = Record.load_from_file(dataset, ratio2version[remove_ratio], 'hedgecut',
                                                                self.n_rounds)
                        logging.debug(f"Done loading, calculating Hellinger distance")
                        model_diff_hedgecut = ModelDiffSingle(dataset, self.n_trees, remove_ratio, self.n_rounds,
                                                              self.keyword, n_jobs=self.n_jobs, record=record_hedgecut)
                        accs_hedgecut = model_diff_hedgecut.get_accuracy()
                        acc_hedgecut_origin = f"{accs_hedgecut[0][0]:.4f}\\textpm {accs_hedgecut[0][1]:.4f}"
                        acc_hedgecut_forget = f"{accs_hedgecut[1][0]:.4f}\\textpm {accs_hedgecut[1][1]:.4f}"
                        acc_hedgecut_retrain = f"{accs_hedgecut[2][0]:.4f}\\textpm {accs_hedgecut[2][1]:.4f}"
                        logging.info(f"{dataset} {remove_ratio} hedgecut done.")
                    else:
                        logging.debug(f"Loading hedgecut results from cache")
                        acc_hedgecut_origin = self.accuracy_table[i * 3, j * 3 + 1]
                        acc_hedgecut_forget = self.accuracy_table[i * 3 + 1, j * 3 + 1]
                        acc_hedgecut_retrain = self.accuracy_table[i * 3 + 2, j * 3 + 1]
                else:
                    acc_dart_origin = acc_dart_forget = acc_dart_retrain = '-'
                    acc_hedgecut_origin = acc_hedgecut_forget = acc_hedgecut_retrain = '-'
                    logging.info(f"{dataset} {remove_ratio} HedgeCut and DART skipped.")

                if _update_deltaboost and (update_datasets is None or dataset in update_datasets):
                    try:
                        logging.debug(f"Loading deltaboost data from output")
                        deltaboost_original_scores = np.zeros(self.n_rounds)
                        deltaboost_deleted_scores = np.zeros(self.n_rounds)
                        deltaboost_retrain_scores = np.zeros(self.n_rounds)
                        keyword2id = {'delete': 0, 'remain': 1, 'test': 2}
                        for t in range(self.n_rounds):
                            try:
                                original_out_path = os.path.join(self.deltaboost_out_path,
                                                                 f"{dataset}_deltaboost_{remove_ratio}_{t}.out")
                                retrain_out_path = os.path.join(self.deltaboost_out_path,
                                                                f"{dataset}_deltaboost_{remove_ratio}_retrain_{t}.out")
                                _, raw_original_scores = get_scores_from_file(original_out_path, out_fmt='float')
                                _, raw_retrain_scores = get_scores_from_file(retrain_out_path, out_fmt='float')

                                if len(raw_original_scores) != 0:
                                    deltaboost_original_scores[t] = raw_original_scores[keyword2id[self.keyword]]
                                    deltaboost_deleted_scores[t] = raw_original_scores[keyword2id[self.keyword] + 3]
                                else:
                                    deltaboost_original_scores[t] = np.nan
                                    warnings.warn(f"Empty original scores for {dataset} {remove_ratio} {t}")

                                if len(raw_retrain_scores) != 0:
                                    deltaboost_retrain_scores[t] = raw_retrain_scores[keyword2id[self.keyword]]
                                else:
                                    deltaboost_retrain_scores[t] = np.nan
                                    warnings.warn(f"Empty retrain scores for {dataset} {remove_ratio} {t}")
                            except IndexError:
                                deltaboost_original_scores[t] = np.nan
                                deltaboost_retrain_scores[t] = np.nan
                                warnings.warn(f"Nan Value scores for {dataset} {remove_ratio} {t}")

                        acc_deltaboost_origin = f"{np.nanmean(deltaboost_original_scores):.4f}\\textpm {np.nanstd(deltaboost_original_scores):.4f}"
                        acc_deltaboost_forget = f"{np.nanmean(deltaboost_deleted_scores):.4f}\\textpm {np.nanstd(deltaboost_deleted_scores):.4f}"
                        acc_deltaboost_retrain = f"{np.nanmean(deltaboost_retrain_scores):.4f}\\textpm {np.nanstd(deltaboost_retrain_scores):.4f}"
                        logging.info(f"Done.")
                    except Exception as e:
                        logging.error(f"Failed to load deltaboost data from output, {e}")
                        acc_deltaboost_origin = acc_deltaboost_forget = acc_deltaboost_retrain = '-'
                else:
                    logging.debug(f"Loading deltaboost results from cache")
                    acc_deltaboost_origin = self.accuracy_table[i * 3, j * 3 + 2]
                    acc_deltaboost_forget = self.accuracy_table[i * 3 + 1, j * 3 + 2]
                    acc_deltaboost_retrain = self.accuracy_table[i * 3 + 2, j * 3 + 2]

                self.accuracy_table[3 * i, 3 * j] = acc_dart_origin
                self.accuracy_table[3 * i + 1, 3 * j] = acc_dart_forget
                self.accuracy_table[3 * i + 2, 3 * j] = acc_dart_retrain
                self.accuracy_table[3 * i, 3 * j + 1] = acc_hedgecut_origin
                self.accuracy_table[3 * i + 1, 3 * j + 1] = acc_hedgecut_forget
                self.accuracy_table[3 * i + 2, 3 * j + 1] = acc_hedgecut_retrain
                self.accuracy_table[3 * i, 3 * j + 2] = acc_deltaboost_origin
                self.accuracy_table[3 * i + 1, 3 * j + 2] = acc_deltaboost_forget
                self.accuracy_table[3 * i + 2, 3 * j + 2] = acc_deltaboost_retrain

                logging.info(f"{dataset}, {remove_ratio} done.")

        if save_path is not None:
            np.savetxt(save_path, self.accuracy_table, fmt='%s', delimiter=',')
            logging.info(f"Table saved to {save_path}")

    def print_latex_forget(self):
        dataset_str = ['\\multirow{2}{*}{%s}' % dataset for dataset in self.datasets]
        table_with_title = np.concatenate([
            np.array(list(zip(dataset_str, np.array([''] * len(dataset_str))))).reshape(-1, 1),
            np.array(
                [r"$H^2(M_r,M;\mathbf{D}_{test})$", r"$H^2(M_r,M_d;\mathbf{D}_{test})$"] * len(self.datasets)).reshape(
                -1, 1),
            self.table_data.astype('U32')], axis=1)
        table_df = pd.DataFrame(table_with_title)
        table_latex = table_df.to_latex(index=False, header=False, escape=False)

        # insert \midrule every two rows
        lines = table_latex.splitlines()
        i = 2 + 2  # first two lines are header
        while i < len(lines) - 2:
            lines.insert(i, '\midrule')
            i += 3
        print('\n'.join(lines))

    def print_latex_accuracy(self):
        # print accuracy latex table
        dataset_str = ['\\multirow{3}{*}{%s}' % dataset for dataset in self.datasets]
        table_with_title = np.concatenate([
            np.array(
                list(zip(dataset_str, np.array([''] * len(dataset_str)), np.array([''] * len(dataset_str))))).reshape(
                -1, 1),
            np.array([r"$M$", r"$M_{d}$", r"$M_{r}$"] * len(self.datasets)).reshape(-1, 1),
            self.accuracy_table], axis=1)
        table_df = pd.DataFrame(table_with_title)
        table_latex = table_df.to_latex(index=False, header=False, escape=False)

        # insert \midrule every three rows
        lines = table_latex.splitlines()
        i = 2 + 3  # first two lines are header
        while i < len(lines) - 2:
            lines.insert(i, '\midrule')
            i += 4
        print('\n'.join(lines))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--n-trees', type=int, default=1)
    parser.add_argument('-acc', action='store_true', default=False)
    parser.add_argument('-p', '--figure-path', type=str, default=None)
    args = parser.parse_args()

    datasets = ['codrna', 'covtype', 'gisette', 'cadata', 'msd']
    # datasets = ['cadata', 'codrna']
    remove_ratios = ['1e-03', '1e-02']
    # remove_ratios = ['0.001', '0.01']
    # plot_score_before_after_removal("../out/remove_test/tree50", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree30", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree10", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree1", datasets, remove_ratios)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=50)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=30)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=10)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=1)
    # print_model_diff(datasets, 1, remove_ratios)
    # print_model_diff(datasets, 10, remove_ratios)
    # print_model_diff(datasets, 30, remove_ratios)
    # print_model_diff(datasets, 50, remove_ratios)

    # for dataset in datasets:
    #     model_diff1 = ModelDiff(f"{dataset}", 1, '1e-03', 100)
    #     model_diff1.predict_(1)
    #     df1 = model_diff1.get_helliger_distance_df(1)
    #     model_diff2 = ModelDiff(f"{dataset}", 1, '1e-02', 100)
    #     model_diff2.predict_(1)
    #     df2 = model_diff2.get_helliger_distance_df(1)
    #     df_combine = pd.concat([df1, df2.drop(columns=df2.columns[0])], axis=1)
    #     print(df_combine.to_latex(escape=False))

    # model_diff = ModelDiff(datasets, remove_ratios, 1, n_rounds=100, n_jobs=1,
    #                        hedgecut_path="/data/junhui/Hedgecut",
    #                        dart_path="/data/junhui/DART",
    #                        deltaboost_out_path="../out/remove_test/tree1/",
    #                        deltaboost_path="../cache_local",
    #                        # forget_table_cache_path="out/forget_table.csv",
    #                        forget_table_cache_path="out/forget_table_susy.csv",
    #                        # accuracy_table_cache_path="out/accuracy_table.csv",
    #                        accuracy_table_cache_path="out/accuracy_table_susy.csv",
    #                        )
    # deltaboost_path="/data/zhaomin/DeltaBoost/cache")
    # model_diff.get_raw_data_forget_(n_bins=50,
    #                                 update_deltaboost=True,
    #                                 update_datasets=['susy'],
    #                                 # save_path="out/forget_table_susy.csv",
    #                                 )
    # model_diff.print_latex_forget()

    # model_diff.get_raw_data_accuracy_(update_deltaboost=False,
    #                                   update_dart=False,
    #                                   update_hedgecut=True,
    #                                   update_datasets=['susy'],
    #                                   # save_path="out/accuracy_table_susy.csv",
    #                                   )
    # model_diff.print_latex_accuracy()
    if args.acc:
        if args.figure_path is None:
            figure_path = f"../fig/acc-tree{args.n_trees}.jpg"
        else:
            figure_path = args.figure_path
        plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=args.n_trees, save_path=figure_path)
    else:
        n_trees = args.n_trees
        model_diff = ModelDiff(datasets, remove_ratios, n_trees, n_rounds=100, n_jobs=1,
                               # hedgecut_path="/data/junhui/Hedgecut",
                               # dart_path="/data/junhui/DART",
                               deltaboost_out_path=f"../out/remove_test/tree{n_trees}/",
                               deltaboost_path="../cache",
                               # forget_table_cache_path=f"../out/forget_table_tree{n_trees}.csv",
                               # forget_table_cache_path=None,
                               # accuracy_table_cache_path=f"../out/accuracy_table_tree{n_trees}.csv",
                               # accuracy_table_cache_path=None,
                               )
        model_diff.get_raw_data_forget_(n_bins=50,
                                        update_dart=False,
                                        update_hedgecut=False,
                                        update_deltaboost=True,
                                        # update_datasets=['msd'],
                                        save_path=f"../out/forget_table_tree{n_trees}.csv",
                                        )
        model_diff.print_latex_forget()


        model_diff.get_raw_data_accuracy_(update_dart=False,
                                          update_hedgecut=False,
                                          update_deltaboost=True,
                                          # update_datasets=['msd'],
                                          save_path=f"../out/accuracy_table_tree{n_trees}.csv",
                                          )
        model_diff.print_latex_accuracy()
