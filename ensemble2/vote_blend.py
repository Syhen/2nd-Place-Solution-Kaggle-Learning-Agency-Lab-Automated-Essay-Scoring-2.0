"""
@created by: heyao
@created at: 2024-05-06 11:43:41
"""
from collections import Counter

import numpy as np
import pandas as pd
import catboost as cat
import scipy as sp
import xgboost as xgb
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from nltk import word_tokenize, sent_tokenize
from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm

from ensemble.utils import load_prediction, load_prediction_from_path
from aes2.utils.metrics import competition_score, OptimizedRounder

tqdm.pandas()


def print_qwk(df, name):
    opt = OptimizedRounder()
    opt.fit(df["raw_pred"], df["score"])
    # , coef=np.array([1.6, 2.55, 3.45, 4.2, 5.1])
    pred = opt.predict(df["raw_pred"])
    print(f"<<< {name} best:", competition_score(pred, df["score"]))
    print(f"<<< {name} best KO:", competition_score(pred[df[df["is_pc2"] == 0].index],
                                                    df["score"].values[df[df["is_pc2"] == 0].index]))
    # print(f"<<< {name} round:", competition_score(df["raw_pred"].values.round(0).astype(int), df["score"]))


df_exp302_large_cope = load_prediction_from_path("../ensemble2/exp302_large_cope")
df_exp306b_large_clean_last = load_prediction_from_path("../ensemble2/exp306_clean")
df_exp306b_large_last = load_prediction_from_path("../ensemble2/exp306_large")
df_exp321 = load_prediction_from_path("../ensemble2/exp321_large")
df_exp320b = load_prediction_from_path("../ensemble2/exp320b_large")

df = pd.read_csv("../my_datasets/learning-agency-lab-automated-essay-scoring-2/train.csv")
df_folds = pd.read_csv("../my_datasets/folds/folds4.csv")
df = df.merge(df_folds, on="essay_id", how="left")
if "is_pc2" not in df.columns:
    df = df.merge(pd.read_csv("../my_datasets/is_pc2.csv"), on="essay_id", how="left")
df["n_words"] = df["full_text"].apply(lambda x: len(x.strip().split()))
df = df.merge(df_exp302_large_cope[["essay_id", "raw_pred"]].rename(
    columns={"raw_pred": "exp302_large_cope"}), on="essay_id", how="left")
df = df.merge(df_exp306b_large_clean_last[["essay_id", "raw_pred"]].rename(
    columns={"raw_pred": "exp306b_large_last_clean"}), on="essay_id", how="left")
df = df.merge(df_exp306b_large_last[["essay_id", "raw_pred"]].rename(
    columns={"raw_pred": "exp306b_large_last"}), on="essay_id", how="left")
df = df.merge(df_exp321[["essay_id", "raw_pred"]].rename(
    columns={"raw_pred": "exp321_large"}), on="essay_id", how="left")
df = df.merge(df_exp320b[["essay_id", "raw_pred"]].rename(
    columns={"raw_pred": "exp320b_large"}), on="essay_id", how="left")
sub = df[["essay_id", "score", "is_pc2"]].copy()

print(df.columns)
columns = [
    "exp302_large_cope", "exp320b_large", "exp306b_large_last", "exp306b_large_last_clean",
    "exp321_large"
]

def find_best_weight(exp, opt, column, idx=-1):
    ths, scores, ko_scores = [], [], []
    w = opt._coef[idx]
    for i in range(-50, 100):
        th = w - i / 100
        opt._coef[idx] = th
        ths.append(th)
        exp["pred2"] = opt.predict(exp[column])
        scores.append(competition_score(exp["score"], exp["pred2"]))
        ko_scores.append(competition_score(exp[exp["is_pc2"] == 0]["score"], exp[exp["is_pc2"] == 0]["pred2"]))
    # print(np.argmax(scores))
    return ths[np.argmax(scores)]


for column in columns:
    print(column)
    opt = OptimizedRounder().fit(df[column], df["score"])
    best_weight = find_best_weight(df, opt, column)
    opt._coef[-1] = best_weight
    print(opt._coef.tolist())
    if "305" in column:
        opt._coef[-1] = 4.9
    # if "exp306" in column:
    #     opt._coef = np.array([1.6267456036469254, 2.792639960297426, 3.6155515754473564, 4.14040939683664, 4.894691388309797])
    sub[column] = opt.predict(df[column]).astype(int)
    print(competition_score(sub["score"], sub[column]))
    print(competition_score(sub[sub["is_pc2"] == 0]["score"], sub[sub["is_pc2"] == 0][column]))


def predict(row):
    most_common = Counter([row[col] for col in columns]).most_common(1)[0]
    if most_common[1] > 1:
        return most_common[0]
    return row["exp302_large_cope"]


sub["pred"] = sub.apply(predict, axis=1)
print(competition_score(sub["pred"], sub["score"]))
print(competition_score(sub[sub["is_pc2"] == 0]["pred"], sub[sub["is_pc2"] == 0]["score"]))
# sub.to_csv("./output/ranking_302_306.csv", index=False)
print(sub["pred"].value_counts(normalize=True))
print(sub[sub["is_pc2"] == 0]["pred"].value_counts(normalize=True))
