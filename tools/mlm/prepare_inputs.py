"""
@created by: heyao
@created at: 2024-05-02 23:54:05
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("../../my_datasets/learning-agency-lab-automated-essay-scoring-2/train.csv")
kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(df, df["score"]):
    break
df_train = df.loc[train_idx]
df_val = df.loc[val_idx]
with open("train.json", "w") as f:
    for line in df_train.full_text:
        f.write(json.dumps({"text": line}) + "\n")
with open("val.json", "w") as f:
    for line in df_val.full_text:
        f.write(json.dumps({"text": line}) + "\n")
