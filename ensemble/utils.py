"""
@created by: heyao
@created at: 2024-05-06 11:43:51
"""
import os
from glob import glob

import pandas as pd


def load_prediction(path_regex):
    filenames = glob(path_regex)
    filenames = list(sorted(filenames))
    print(f"<<< loading {filenames}")
    df = pd.concat([pd.read_csv(filename) for filename in filenames], axis=0).reset_index(drop=True)
    if "is_pc2" not in df.columns:
        df = df.merge(pd.read_csv("../my_datasets/is_pc2.csv"), on="essay_id", how="left")
    return df


def load_prediction_from_path(path):
    filenames = os.listdir(path)
    filenames = [i for i in filenames if i.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(path, filename))
                    for filename in filenames], axis=0).reset_index(drop=True)
    if "is_pc2" not in df.columns:
        df = df.merge(pd.read_csv("../my_datasets/is_pc2.csv"), on="essay_id", how="left")
    return df
