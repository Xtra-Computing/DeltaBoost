import numpy as np
import argparse
import re
import os
import sys
parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--datasets', nargs='+', type=str)
# parser.add_argument('-r', '--rate', type=float)
parser.add_argument('-t', '--tree', type=int)
# parser.add_argument('--is-retrain', action='store_true')

args = parser.parse_args()


def load_file(file_base_path, ratio: str, n_runs):
    removing_time = [0 for _ in range(n_runs)]
    training_time = [0 for _ in range(n_runs)]
    for i in range(n_runs):
        fn = f"{file_base_path}_deltaboost_{ratio}_{i}.out"
        # print("Loading file: ", fn)
        with open(fn, "r") as f:
            for line in f:
                if "removing time" in line:
                    removing_time[i] = float(re.findall("\d+\.\d+", line)[0])
                if "training time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
                if "Init booster time" in line:
                    training_time[i] += float(re.findall("\d+\.\d+", line)[0])
    db_train_time_str = rf"{np.mean(training_time): .3f} \textpm {np.std(training_time):.3f}"
    db_remove_time_str = rf"{np.mean(removing_time): .3f} \textpm {np.std(removing_time):.3f}"

    fn_gbdt = f"{file_base_path}_gbdt_{ratio}.out"
    gbdt_time = 0
    # print("Loading file: ", fn_gbdt)
    with open(fn_gbdt, "r") as f:
        for line in f:
            if "training time" in line:
                gbdt_time += float(re.findall("\d+\.\d+", line)[0])
            if "Init booster time" in line:
                gbdt_time += float(re.findall("\d+\.\d+", line)[0])
    gbdt_time_str = rf"{gbdt_time: .3f}"

    speedup = gbdt_time / np.mean(removing_time)

    print(f"{gbdt_time_str}\t& {db_train_time_str}\t & {db_remove_time_str}\t & {speedup:.2f}x \\\\")



if __name__ == '__main__':
    datasets = ['codrna', 'covtype', 'gisette', 'cadata', 'msd']
    print("Thunder\t& DB-Train\t& DB-Remove\t& Speedup (Thunder) \\\\")
    for dataset in datasets:
        load_file(f"../out/remove_test/tree{args.tree}/"+dataset, "1e-03", 100)
        load_file(f"../out/remove_test/tree{args.tree}/"+dataset, "1e-02", 100)


