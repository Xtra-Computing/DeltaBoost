import argparse

import numpy as np

from train_test_split import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('-if', '--input-fmt', type=str, default='libsvm')
    parser.add_argument('-of', '--output-fmt', type=str, default='csv')
    args = parser.parse_args()

    X, y = load_data(data_path=args.data_path, data_fmt=args.input_fmt, scale_y=False, output_dense=True)
    y = (y - 1920) / 100
    save_data(X, y, save_path=args.data_path, save_fmt=args.output_fmt)
