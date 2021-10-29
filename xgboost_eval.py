# %%
import os
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import KFold, cross_val_score

import utils

# how to use this script
# python xgboost_eval.py /data/UKBB/capture24_neurips/prepared_data
# 11 classes Tasks
# python xgboost_eval.py /data/UKBB/capture24_neurips/prepared_data --label=label:WillettsSpecific2018

# Best parameters:
# {'colsample_bytree': 0.507136631991765, 'gamma': 5.536187121942828, 'max_depth': 7.0,
# 'min_child_weight': 0.0, 'reg_alpha': 40.0, 'reg_lambda': 0.6837864513214184}

#{'colsample_bytree': 0.6802307534371619, 'gamma': 1.0340769559639673,
# 'max_depth': 13.0, 'min_child_weight': 0.0, 'n_estimators': 280.0, 'reg_alpha': 51.0, 'reg_lambda': 0.5719966397769551}


def main(args):
    # For reproducibility
    np.random.seed(42)

    data = pd.read_pickle(os.path.join(args.datadir, 'featframe.pkl'))
    featcols = np.loadtxt(os.path.join(args.datadir, 'features.txt'), dtype='str')
    labelcol = args.label
    print("Data loaded")

    if args.smoke_test:
        data = data.sample(frac=0.1, random_state=42)

    # Use P001-P100 as derivation set and the rest as test set
    data_deriv = data[data['pid'].str.contains('P0[0-9][0-9]|P100')]
    data_test = data[~data['pid'].str.contains('P0[0-9][0-9]|P100')]

    X = data_deriv[featcols].to_numpy()
    Y = data_deriv[labelcol].to_numpy()
    print("Label class: ", len(np.unique(Y)))

    X_test = data_test[featcols].to_numpy()
    Y_test = data_test[labelcol].to_numpy()

    if args.opt:
        num_folds = 3
        kf = KFold(n_splits=num_folds)

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

        def objective(space):
            clf = XGBClassifier(
                n_estimators=int(space['n_estimators']), max_depth=int(space['max_depth']), gamma=space['gamma'],
                reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
                reg_lambda=int(space['reg_lambda']),
                colsample_bytree=int(space['colsample_bytree']))

            error = -cross_val_score(clf, X_train, y_train,
                             cv=kf, scoring="neg_log_loss", n_jobs=-1).mean()

            print ("SCORE:", error)
            return {'loss':  error, 'status': STATUS_OK}

        space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': hp.quniform('n_estimators', 60, 300, 5),
                 'seed': 0
                 }

        trials = Trials()

        best_hyperparams = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)
    else:
        print("Start training")
        clf = XGBClassifier(
            max_depth=13, reg_alpha=51, reg_lambda=0.572, min_child_weight=0, gamma=1.03,
            colsample_bytree=int(0.6802307534371619), n_estimators=250)
        # clf = XGBClassifier()
        clf.fit(X, Y)

        Y_pred = clf.predict(X_test)

        print("XGBoost performance:")
        utils.metrics_report(Y_test, Y_pred, n_jobs=args.n_jobs)


        Y_prob = clf.predict_proba(X)
        hmm_params = utils.train_hmm(Y_prob, Y)
        Y_pred_hmm = utils.viterbi(Y_pred, hmm_params)

        print("XGBoost + HMM performance:")
        utils.metrics_report(Y_test, Y_pred_hmm, n_jobs=args.n_jobs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--label', default='label:Willetts2018')
    parser.add_argument('--n_estimators', type=int, default=3000)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--opt', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
