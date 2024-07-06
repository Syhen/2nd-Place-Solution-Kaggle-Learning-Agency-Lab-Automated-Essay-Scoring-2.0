"""
@created by: heyao
@created at: 2024-04-30 00:57:50
"""
import pandas as pd


def sample_data_by_dist(df_train):
    df_new = df_train[df_train["is_pc2"] == 0]
    df_pc2 = df_train[df_train["is_pc2"] == 1]
    label_dist = df_new["score"].value_counts()
    mapping = dict(zip(label_dist.index, label_dist.values))

    dist_pc2 = df_pc2["score"].value_counts()
    mapping_pc2 = dict(zip(dist_pc2.index, dist_pc2.values))
    pc2_counts = {}
    pc2_counts[4] = mapping_pc2[4]
    for i in [1, 2, 3, 5, 6]:
        pc2_counts[i] = int(mapping[i] / mapping[4] * mapping_pc2[4])
    temps = []
    for score, temp in df_pc2.groupby("score"):
        temps.append(temp.sample(pc2_counts[score]))
    return pd.concat(temps + [df_new], axis=0).reset_index(drop=True)


if __name__ == '__main__':
    df = pd.read_csv("/Users/heyao/projects/kaggle-aes2/my_datasets/learning-agency-lab-automated-essay-scoring-2/train.csv")
    df = df.merge(
        pd.read_csv("/Users/heyao/projects/kaggle-aes2/my_datasets/is_pc2.csv"),
        on="essay_id", how="left"
    )
    print(df.shape)
    new_sampled = sample_data_by_dist(df)
    print(new_sampled.shape)
