# %%
import os
import argparse
import numpy as np
import pandas as pd
from xgboost_eval import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import utils

# how to use this script
# python lr.py /data/UKBB/capture24_neurips/prepared_data

def main(args):
    # For reproducibility
    rand_seed = 42
    np.random.seed(42)

    data = pd.read_pickle(os.path.join(args.datadir, 'featframe.pkl'))
    featcols = np.loadtxt(os.path.join(args.datadir, 'features.txt'), dtype='str')
    labelcol = args.label
    print("Data loaded")


    # Use P001-P100 as derivation set and the rest as test set
    data_deriv = data[data['pid'].str.contains('P0[0-9][0-9]|P100')]
    data_test = data[~data['pid'].str.contains('P0[0-9][0-9]|P100')]

    X = data_deriv[featcols].to_numpy()
    Y = data_deriv[labelcol].to_numpy()

    X_test = data_test[featcols].to_numpy()
    Y_test = data_test[labelcol].to_numpy()

    # X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = LogisticRegression(solver='lbfgs', max_iter=5000,
                             random_state=rand_seed).fit(X, Y)

    Y_pred = clf.predict(X_test)

    print("RF performance:")
    utils.metrics_report(Y_test, Y_pred, n_jobs=args.n_jobs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--label', default='label:Willetts2018')
    parser.add_argument('--n_estimators', type=int, default=3000)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    main(args)
