"""
@created by: heyao
@created at: 2024-04-24 01:43:30
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("./my_datasets/learning-agency-lab-automated-essay-scoring-2/train.csv")
print(df.head(1).T)
print(df["score"].value_counts().sort_index())
df["score"].hist()
plt.show()
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(kfold.split(df, y=df["score"])):
    df.loc[val_idx, "kfold"] = i
os.system("mkdir ./my_datasets/folds")
df[["essay_id", "kfold"]].to_csv("./my_datasets/folds/folds4.csv", index=False)
