"""
@created by: heyao
@created at: 2022-09-05 20:55:34
"""
import random
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import scipy
import scipy as sp
from sklearn import metrics


def competition_score(y_pred, y_true):
    return metrics.cohen_kappa_score(y_pred, y_true, weights="quadratic")


def sample_to_distribution(df, target_distribution=None, sample_ratio=0.6):
    target_distribution = {
        3: 0.362859,
        2: 0.272895,
        4: 0.226845,
        1: 0.072341,
        5: 0.056047
    } if target_distribution is None else target_distribution
    df = df.copy()
    df_val_1_5 = df[df.score != 6]
    df_val_6 = df[(df.score == 6) & (df["is_pc2"] == 0)]
    current_count = df_val_1_5[df_val_1_5["is_pc2"] == 0]["score"].value_counts()
    total = len(df_val_1_5[df_val_1_5["is_pc2"] == 0])
    sample_counts = {k: min(int(target_distribution[k] * total * sample_ratio), current_count[k]) for k in range(1, 6)}
    sampled = []
    for score, count in sample_counts.items():
        count = min(count + random.randint(-5, 5), count)
        sampled.append(df_val_1_5[(df_val_1_5["is_pc2"] == 0) & (df_val_1_5["score"] == score)].sample(count))
    n_6 = min(len(df_val_6), int(total * 0.01))
    sampled.append(df_val_6.sample(n_6))
    df_val2 = pd.concat(sampled, axis=0).reset_index(drop=True)
    return df_val2


def upsample_to_distribution(df, target_distribution=None):
    target_distribution = {
        3: 0.362859,
        2: 0.272895,
        4: 0.226845,
        1: 0.072341,
        5: 0.056047
    } if target_distribution is None else target_distribution
    df = df.copy()
    df_val_1_5 = df[df.score != 6]
    df_val_6 = df[(df.score == 6) & (df["is_pc2"] == 0)]
    current_count = df_val_1_5[df_val_1_5["is_pc2"] == 0]["score"].value_counts()
    total = len(df_val_1_5[df_val_1_5["is_pc2"] == 0])
    sample_counts = {k: min(int(target_distribution[k] * total), current_count[k]) for k in range(1, 6)}
    sampled = []
    for score, count in sample_counts.items():
        count = min(count + random.randint(-5, 5), count)
        sampled.append(df_val_1_5[(df_val_1_5["is_pc2"] == 0) & (df_val_1_5["score"] == score)].sample(count, replace=True))
    n_6 = min(len(df_val_6), int(total * 0.01))
    sampled.append(df_val_6.sample(n_6))
    df_val2 = pd.concat(sampled, axis=0).reset_index(drop=True)
    return df_val2


class OptimizedRounder(object):
    def __init__(self, init_coef=None, alpha=0.5, mean_score=False):
        self.coef_ = 0
        self._coef = None
        self.init_coef = init_coef if init_coef is not None else [1.5, 2.5, 3.5, 4.5, 5.5]
        self.alpha = alpha
        self.mean_score = mean_score

    def _kappa_loss(self, coef, X, y, index=None):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[1, 2, 3, 4, 5, 6])
        if index is None:
            return -metrics.cohen_kappa_score(y, preds, weights='quadratic')
        else:
            return -(metrics.cohen_kappa_score(y[index], preds[index], weights='quadratic') * self.alpha +
                     metrics.cohen_kappa_score(y[~index], preds[~index], weights='quadratic') * (1 - self.alpha))

    def fit(self, X, y, index=None):
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        loss_partial = partial(self._kappa_loss, X=X, y=y, index=index)
        initial_coef = np.array(self.init_coef)
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        self._coef = self.coef_['x']
        return self

    def predict(self, X, coef=None):
        coef = coef if coef is not None else self._coef
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[1, 2, 3, 4, 5, 6])
        return preds

    def coefficients(self):
        return self.coef_['x']
