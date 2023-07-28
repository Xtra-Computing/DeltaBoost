import os.path
import re
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ujson as json
from joblib import Parallel, delayed
from tqdm import tqdm
from joblib import Memory

from GBDT import GBDT
from train_test_split import load_data
from plot_results import ModelDiffSingle

# logging.basicConfig(
#     format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%Y-%m-%d:%H:%M:%S',
#     level=logging.INFO)
memory = Memory(location="cache")


@memory.cache
def _hsr_get_single_diff(hsr, dataset, n_jobs, n_trees, remove_ratio, n_rounds, infer_data, cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"hsr{hsr}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=n_trees, remove_ratio=remove_ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=50, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete


def plot_bagging_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                        cache_dir="../cache/ablation-bagging/"):
    forget_data = []
    hsr_list = [1, 3, 5, 7, 10]

    for hsr in hsr_list:
        original_vs_retrain, retrain_vs_delete = _hsr_get_single_diff(hsr, dataset, n_jobs, n_trees, remove_ratio,
                                                                      n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(f"hsr {hsr} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    x = hsr_list
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("Sampling cycle (1 / sampling rate)")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(f"Bagging")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-bagging-{dataset}.jpg")
    # plt.show()


@memory.cache
def _iteration_get_single_diff(iteration, dataset, n_jobs, remove_ratio, n_rounds, infer_data, cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"iter{iteration}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=iteration, remove_ratio=remove_ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=50, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete


def plot_iteration_forget(dataset, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                          cache_dir="../cache/ablation-iter/"):
    forget_data = []
    iteration_list = [1, 5, 10, 50, 100]

    for iteration in iteration_list:
        original_vs_retrain, retrain_vs_delete = _iteration_get_single_diff(iteration, dataset, n_jobs, remove_ratio,
                                                                            n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(f"iteration {iteration} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    x = iteration_list
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(f"#iterations")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-iter-{dataset}.jpg")


@memory.cache
def _quantization_get_single_diff(quantization, dataset, n_jobs, n_trees, remove_ratio, n_rounds, infer_data,
                                  cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"quan{quantization}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=n_trees, remove_ratio=remove_ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=50, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete


def plot_quantization_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                             cache_dir="../cache/ablation-quantization/"):
    forget_data = []
    quantization_list = [2, 4, 8, 16, 32]

    for quantization in quantization_list:
        original_vs_retrain, retrain_vs_delete = _quantization_get_single_diff(quantization, dataset, n_jobs, n_trees,
                                                                               remove_ratio,
                                                                               n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(
            f"quantization {quantization} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    x = quantization_list
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("Quantized interval")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(f"Quantization")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-quan-{dataset}.jpg")


@memory.cache
def _nbins_get_single_diff(n_bins, dataset, n_jobs, n_trees, remove_ratio, n_rounds, infer_data, cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"nbins{n_bins}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=n_trees, remove_ratio=remove_ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=n_bins, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete


def plot_nbins_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                      cache_dir="../cache/ablation-nbins/"):
    forget_data = []
    nbins_list = [50, 100, 300, 500, 1000][::-1]

    for nbins in nbins_list:
        original_vs_retrain, retrain_vs_delete = _nbins_get_single_diff(nbins, dataset, n_jobs, n_trees, remove_ratio,
                                                                        n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(f"nbins {nbins} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    x = nbins_list
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("Max bin size $t$")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(f"Number of bins")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-nbins-{dataset}.jpg")


@memory.cache
def _regularization_get_single_diff(regularization, dataset, n_jobs, n_trees, remove_ratio, n_rounds, infer_data,
                                    cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"reg{regularization}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=n_trees, remove_ratio=remove_ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=50, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete


def plot_regularization_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                               cache_dir="../cache/ablation-regularization/"):
    forget_data = []
    regularization_list = [0, 1, 3, 5, 10]

    for regularization in regularization_list:
        original_vs_retrain, retrain_vs_delete = _regularization_get_single_diff(regularization, dataset, n_jobs,
                                                                                 n_trees, remove_ratio,
                                                                                 n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(
            f"regularization {regularization} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    x = regularization_list
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(rf"Regularization")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-reg-{dataset}.jpg")





@memory.cache
def _ratio_get_single_diff(ratio, dataset, n_jobs, n_trees, remove_ratio, n_rounds, infer_data, cache_dir):
    deltaboost_path = os.path.join(cache_dir, f"ratio{ratio}")
    model_diff = ModelDiffSingle(dataset=dataset, n_trees=n_trees, remove_ratio=ratio, n_jobs=n_jobs,
                                 n_rounds=n_rounds, keyword=infer_data, deltaboost_path=deltaboost_path)
    _original_vs_retrain, _retrain_vs_delete = model_diff.get_hellinger_distance(n_bins=50, return_std=True)
    return _original_vs_retrain, _retrain_vs_delete

def plot_ratio_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data='test', n_jobs=1,
                      cache_dir="../cache/ablation-ratio/"):
    forget_data = []
    ratio_list = list(reversed(["5e-01", "2e-01", "1e-01", "5e-02", "1e-02", "1e-03"]))

    for ratio in ratio_list:
        original_vs_retrain, retrain_vs_delete = _ratio_get_single_diff(ratio, dataset, n_jobs, n_trees, remove_ratio,
                                                                        n_rounds, infer_data, cache_dir)
        retrain_vs_delete_mean, retrain_vs_delete_std = retrain_vs_delete
        forget_data.append(retrain_vs_delete)

        print(f"ratio {ratio} loaded. mean: {retrain_vs_delete_mean:.4f}, std: {retrain_vs_delete_std:.4f}")

    # plot the mean and std as error bar with a line connecting them
    print("Plotting")
    fig, ax = plt.subplots()
    # convert ratio to percentage
    x = [f"{float(ratio) * 100:.2g}%" for ratio in ratio_list]
    y = [data[0] for data in forget_data]
    yerr = [data[1] for data in forget_data]
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='k')
    ax.plot(x, y, color='k')
    ax.set_xlabel("$n_d/n$")
    ax.set_ylabel(r"Forgetfulness")
    ax.set_title(rf"Ratio of deletion")
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-ratio-{dataset}.jpg")


def load_time(file_path, n_runs, is_retrain):
    removing_time = [0. for _ in range(n_runs)]
    training_time = [0. for _ in range(n_runs)]
    if is_retrain:
        file_path += '_retrain'
    for i in range(n_runs):
        fn = file_path + "_" + str(i) + ".out"
        print("Loading file: ", fn)
        with open(fn, "r") as f:
            for line in f:
                if "removing time" in line:
                    removing_time[i] = float(re.findall("\d+\.\d+", line)[0])
                if "training time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
                if "Init booster time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
    return np.mean(training_time), np.mean(removing_time)


def plot_ratio_time(dataset, n_trees, n_rounds, out_dir="../out/ablation-ratio/"):
    training_time_list = []
    removing_time_list = []
    ratio_list = list(reversed(["5e-01", "2e-01", "1e-01", "5e-02", "1e-02", "1e-03"]))

    for ratio in ratio_list:
        deltaboost_base_path = os.path.join(out_dir, f"ratio{ratio}", f"{dataset}_deltaboost_{ratio}")
        training_time, removing_time = load_time(deltaboost_base_path, n_rounds, is_retrain=False)
        training_time_list.append(training_time)
        removing_time_list.append(removing_time)

    print("Plotting")
    fig, ax = plt.subplots()
    # convert ratio to percentage
    x = ([f"{float(ratio) * 100:.2g}%" for ratio in ratio_list])
    # ax.plot(x, training_time_list, color='k', marker='o', label="training time")
    ax.plot(x, removing_time_list, color='k', marker='^', label="removing time")

    # plot a threshold line of 0.238 with round dots
    ax.plot(x, [0.238 for _ in range(len(x))], color='k', linestyle='--', marker='o', label="threshold")


    ax.set_xlabel("$n_d/n$")
    ax.set_ylabel(r"Time (s)")
    ax.set_title(rf"Ratio of deletion")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../fig/ablation/ablation-ratio-time-{dataset}.jpg")





if __name__ == '__main__':
    os.makedirs("../fig/ablation", exist_ok=True)

    plt.rcParams.update({'font.size': 18})
    dataset = 'codrna'
    n_trees = 10
    remove_ratio = "1e-03"
    n_rounds = 50
    infer_data = 'test'
    n_jobs = 1

    plot_bagging_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data)
    plot_iteration_forget(dataset, remove_ratio, n_rounds, infer_data)
    plot_quantization_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data)
    plot_regularization_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data)
    plot_nbins_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data)
    plot_ratio_forget(dataset, n_trees, remove_ratio, n_rounds, infer_data)
    plot_ratio_time(dataset, n_trees, n_rounds)
