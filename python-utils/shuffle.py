import argparse

import numpy as np

from train_test_split import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('-if', '--input-fmt', type=str, default='libsvm')
    parser.add_argument('-of', '--output-fmt', type=str, default='csv')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    X, y = load_data(data_path=args.data_path, data_fmt=args.input_fmt, scale_y=False, output_dense=True)
    np.random.seed(args.seed)
    indices = np.arange(np.shape(X)[0])
    np.random.shuffle(indices)
    X = X[indices, :]
    y = y[indices]
    save_data(X, y, save_path=args.data_path, save_fmt=args.output_fmt)
