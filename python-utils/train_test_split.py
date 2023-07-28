import os.path

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import argparse
from pathlib import Path


def load_data(data_path, data_fmt, scale_y=False, output_dense=False, header=None) -> tuple:
    """
    :param output_dense: whether to output dense matrix. If set to false, csr_matrix will be output
    :param scale_y: whether to scale y to [0,1]
    :param data_fmt: data format (e.g. libsvm)
    :param data_path: path of the data
    :return: data, labels
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in ['libsvm', 'csv'], "Unsupported format"

    if data_fmt == 'libsvm':
        X, y = load_svmlight_file(data_path)
        if output_dense:
            X = X.toarray()
        print(f"Got X with shape {X.shape}, y with shape {y.shape}")
    elif data_fmt == 'csv':
        csv_data = pd.read_csv(data_path, header=header).to_numpy()
        X = csv_data[:, 1:]
        y = csv_data[:, 0]
        print(f"Got X with shape {X.shape}, y with shape {y.shape}")
    else:
        assert False

    if scale_y:
        print("Scaling y to [0,1]")
        y = MinMaxScaler((0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1)

    return X, y


def split_data(data, labels, val_rate=0.1, test_rate=0.2, seed=0):
    print("Splitting...")
    indices = np.arange(data.shape[0])
    if np.isclose(val_rate, 0.0):
        train_data, test_data, train_labels, test_labels, train_idx, test_idx = \
            train_test_split(data, labels, indices, test_size=test_rate, shuffle=True, random_state=seed)
        return train_data, None, test_data, train_labels, None, test_labels, train_idx, None, test_idx
    elif np.isclose(test_rate, 0.0):
        train_data, val_data, train_labels, val_labels, train_idx, val_idx = \
            train_test_split(data, labels, indices, test_size=val_rate, shuffle=True, random_state=seed)
        return train_data, val_data, None, train_labels, val_labels, None, train_idx, val_idx, None
    else:
        train_val_data, test_data, train_val_labels, test_labels, train_val_idx, test_idx = \
            train_test_split(data, labels, indices, test_size=test_rate, shuffle=True, random_state=seed)
        split_val_rate = val_rate / (1. - test_rate)
        train_data, val_data, train_labels, val_labels, train_idx, val_idx = \
            train_test_split(train_val_data, train_val_labels, train_val_idx, shuffle=True, test_size=split_val_rate,
                             random_state=seed)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx


def save_data(X, y, save_path, save_fmt='csv'):
    data_fmt = save_fmt.lower()
    assert data_fmt in ['libsvm', 'csv'], "Unsupported format"

    if data_fmt == 'libsvm':
        dump_svmlight_file(X, y, save_path, zero_based=False)
    elif data_fmt == 'csv':
        pd.DataFrame(np.concatenate([y.reshape(-1, 1), X], axis=1)).to_csv(save_path, index=None, header=None)
    else:
        assert False

    print("Saved {} data to {}".format(X.shape, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('-if', '--input-fmt', type=str, default='libsvm')
    parser.add_argument('-of', '--output-fmt', type=str, default='csv')
    parser.add_argument('--scale-y', action='store_true')
    parser.add_argument('-v', '--val-rate', type=float, default=0.1)
    parser.add_argument('-t', '--test-rate', type=float, default=0.2)
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    main_path = os.path.splitext(args.data_path)[0]
    X, y = load_data(data_path=main_path, data_fmt=args.input_fmt, scale_y=args.scale_y, output_dense=True)
    train_X, val_X, test_X, train_y, val_y, test_y, _, _, _ = split_data(
        X, y, val_rate=args.val_rate, test_rate=args.test_rate, seed=args.seed)

    save_data(train_X, train_y, save_path=main_path + ".train", save_fmt=args.output_fmt)
    if not np.isclose(args.val_rate, 0):
        save_data(val_X, val_y, save_path=main_path + ".val", save_fmt=args.output_fmt)
    if not np.isclose(args.test_rate, 0):
        save_data(test_X, test_y, save_path=main_path + ".test", save_fmt=args.output_fmt)
