# %%
import os
import argparse
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier

import utils


def main(args):
    # For reproducibility
    np.random.seed(42)

    if args.window_len is not None:
        args.datadir = os.path.join(args.datadir, 'prepared_data_vary_winsize', f'prepared_data_{args.window_len}s')

    data = pd.read_pickle(os.path.join(args.datadir, 'featframe.pkl'))
    featcols = np.loadtxt(os.path.join(args.datadir, 'features.txt'), dtype='str')
    labelcol = args.label

    if args.smoke_test:
        data = data.sample(frac=0.1, random_state=42)

    # Use P001-P100 as derivation set and the rest as test set
    data_deriv = data[data['pid'].str.contains('P0[0-9][0-9]|P100')]
    data_test = data[~data['pid'].str.contains('P0[0-9][0-9]|P100')]

    X = data_deriv[featcols].to_numpy()
    Y = data_deriv[labelcol].to_numpy()
    pid = data_deriv['pid'].to_numpy()

    X_test = data_test[featcols].to_numpy()
    Y_test = data_test[labelcol].to_numpy()
    
    original_X_shape = X.shape
    if args.n_users is not None:
        X, Y, pid = utils.get_data_from_n_users(X, Y, pid, args.n_users)
    if args.n_samples is not None:
        X, Y, pid = utils.get_data_from_n_samples(X, Y, pid, args.n_samples)
    new_X_shape = X.shape
    print(f"X shape from {original_X_shape} --> {new_X_shape}")

    clf = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        replacement=True,
        sampling_strategy='not minority',
        oob_score=True,
        n_jobs=args.n_jobs,
        random_state=42,
    )
    clf.fit(X, Y)

    Y_pred = clf.predict(X_test)

    print("RF performance:")
    rf_results = utils.metrics_report(Y_test, Y_pred, n_jobs=args.n_jobs, prefix='test')

    hmm_params = utils.train_hmm(clf.oob_decision_function_, Y, clf.classes_)
    Y_pred_hmm = utils.viterbi(Y_pred, hmm_params)

    print("RF + HMM performance:")
    rf_hmm_results = utils.metrics_report(Y_test, Y_pred_hmm, n_jobs=args.n_jobs, prefix='test/hmm')

    record = {**rf_results, **rf_hmm_results}
    record['data/n_users'] = args.n_users
    record['data/n_sampels'] = args.n_samples

    if args.window_len is not None:
        record['data/window_len'] = args.window_len
        utils.write_experiment_results_to_csv(record, "/nfs-share/catherine/workspace/capture24/rf_results_windows.csv")
    else:
        utils.write_experiment_results_to_csv(record, "/nfs-share/catherine/workspace/capture24/rf_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/datasets/capture24/data/neurips_data/')
    parser.add_argument('--label', default='label:Willetts2018')
    parser.add_argument('--n_estimators', type=int, default=3000)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--n_users', type=int)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    main(args)
