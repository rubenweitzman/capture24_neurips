import os
import numpy as np
from sklearn import metrics
from joblib import Parallel, delayed
import torch
import torchmetrics
import pandas as pd


def train_hmm(Y_prob, Y_true, labels=None, uniform_prior=True):
    ''' https://en.wikipedia.org/wiki/Hidden_Markov_model '''

    if labels is None:
        labels = np.unique(Y_true)

    if uniform_prior:
        # All labels with equal probability
        prior = np.ones(len(labels)) / len(labels)
    else:
        # Label probability equals empirical rate
        prior = np.mean(Y_true.reshape(-1,1)==labels, axis=0)

    emission = np.vstack(
        [np.mean(Y_prob[Y_true==label], axis=0) for label in labels]
    )
    transition = np.vstack(
        [np.mean(Y_true[1:][(Y_true==label)[:-1]].reshape(-1,1)==labels, axis=0)
            for label in labels]
    )

    params = {'prior':prior, 'emission':emission, 'transition':transition, 'labels':labels}

    return params


def viterbi(Y_obs, hmm_params):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    def log(x):
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y_obs)
    nlabels = len(labels)

    Y_obs = np.where(Y_obs.reshape(-1,1)==labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0,:] = log(prior) + log(emission[:, Y_obs[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j,i] = np.max(
                log(emission[i, Y_obs[j]]) + \
                log(transition[:, i]) + \
                probs[j-1,:])  # probs already in log scale
    viterbi_path = np.zeros_like(Y_obs)
    viterbi_path[-1] = np.argmax(probs[-1,:])
    for j in reversed(range(nobs-1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j+1]]) + \
            probs[j,:])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path


def bootstrapCI(f, sample, nboots=100, n_jobs=4):
    """ Bootstrap confidence intervals
    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    """

    mu = f(sample)

    boots = np.random.choice(sample, size=(nboots, len(sample)))

    if n_jobs != 0:
        mus = Parallel(n_jobs=n_jobs)(delayed(f)(boot) for boot in boots)
    else:
        mus = [f(boot) for boot in boots]
    mus = np.stack(mus)

    mu_hi, mu_low = mu - np.percentile(mus - mu, (2.50, 97.50), axis=0)

    return mu_low, mu_hi


def metrics_report(Y_true, Y_pred, nboots=100, n_jobs=4):

    # Print the cute sklearn report
    print(metrics.classification_report(Y_true, Y_pred, zero_division=0))

    # Compute f1, phi and kappa scores
    # Use bootstrap for confidence intervals

    def f(idxs):
        _Y_true, _Y_pred = Y_true[idxs], Y_pred[idxs]

        f1 = metrics.f1_score(_Y_true, _Y_pred, zero_division=0, average='macro')
        phi = metrics.matthews_corrcoef(_Y_true, _Y_pred)
        kappa = metrics.cohen_kappa_score(_Y_true, _Y_pred)

        return np.asarray([f1, phi, kappa])

    idxs = np.arange(len(Y_true))
    (f1, phi, kappa) = f(idxs)
    (f1_low, phi_low, kappa_low), (f1_hi, phi_hi, kappa_hi) = bootstrapCI(f, idxs, nboots, n_jobs)

    print(f"   f1: {f1:.3f} ({f1_low:.3f}, {f1_hi:.3f})")
    print(f"  phi: {phi:.3f} ({phi_low:.3f}, {phi_hi:.3f})")
    print(f"kappa: {kappa:.3f} ({kappa_low:.3f}, {kappa_hi:.3f})")


def write_experiment_results_to_csv(cfg, record):
    path = cfg.record_path
    
    # populate record
    record['data/n_users'] = cfg.data['n_users']
    record['data/n_samples'] = cfg.data['n_samples']
    record['data/window_len'] = cfg.data['window_len']
    record['model/is_cnnlstm'] = cfg.model['is_cnnlstm']
    record['model/pretrained'] = False

    df = pd.DataFrame([record])
    
    if os.path.exists(path):
        df_all = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df_all.to_csv(path, index=False)  
    else:
        df.to_csv(path, index=False)



def get_data_from_n_users(X, Y, pid, n_users):
    selected_train_users = list(set(pid))
    selected_train_users = np.random.choice(selected_train_users, n_users, replace=False)
    print("selected_train_users:", selected_train_users)
    indices = np.where(np.logical_or.reduce([pid == u for u in selected_train_users]))[0]
    X = X[indices]
    Y = Y[indices]
    pid = pid[indices]
    return X, Y, pid

def get_data_from_n_samples(X, Y, pid, n_samples):
    """
    n_samples per class
    """
    indices = []
    for class_ in set(Y):
        class_indices = np.where(Y == class_)[0]
        indices.append(np.random.choice(class_indices, n_samples, replace=False))
    indices = np.concatenate(indices)
    X = X[indices]
    Y = Y[indices]
    pid = pid[indices]
    return X, Y, pid



def use_shorter_windows(window_length, X, Y, pid):
    """
    cutting original 10-second window into multiple copies
    """
    original = X.shape
    copies = 1000 // window_length
    X = np.concatenate(np.hsplit(X, copies))
    Y = np.concatenate([Y]*copies)
    pid = np.concatenate([pid]*copies)
    print(f'Using {window_length}-second windows: {original} --> {X.shape}')
    return X, Y, pid