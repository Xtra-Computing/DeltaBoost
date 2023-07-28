import numpy as np
import argparse

from train_test_split import load_data, save_data


def remove_sample(X, y, removing_indices: list):
    mask = np.ones(X.shape[0], dtype=bool)
    mask[removing_indices] = False
    X_out = X[mask]
    y_out = y[mask]
    X_remain = X[~mask]
    y_remain = y[~mask]
    return X_out, y_out, X_remain, y_remain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('-if', '--input-fmt', type=str, default='libsvm')
    parser.add_argument('-of', '--output-fmt', type=str, default='csv')
    parser.add_argument('--scale-y', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-r', '--remove_ratio', type=float, default=0)
    args = parser.parse_args()

    X, y = load_data(data_path=args.data_path, data_fmt=args.input_fmt, scale_y=args.scale_y, output_dense=True)
    n_remove = int(args.remove_ratio * X.shape[0])
    X_out, y_out, X_remain, y_remain = remove_sample(X, y, removing_indices=list(range(n_remove)))
    print(f"Removed {n_remove} instances.")
    save_data(X_out, y_out, save_path=f"{args.data_path}.remain_{args.remove_ratio:.0e}", save_fmt=args.output_fmt)
    save_data(X_remain, y_remain, save_path=f"{args.data_path}.delete_{args.remove_ratio:.0e}", save_fmt=args.output_fmt)
