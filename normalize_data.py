import numpy as np
import pandas as pd

def normalize(df):
    columns = [col for col in df.columns if (col != 'label' and col != 'release_id')]
    train_stats = {col: [] for col in columns}
    for col in columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        train_stats[col] = [col_mean, col_std]
        df[col] = (df[col] - col_mean) / col_std
    return train_stats, df


def fit_normalize(train_stats, df):
    for col in train_stats:
        col_mean = train_stats[col][0]
        col_std = train_stats[col][1]
        df[col] = (df[col] - col_mean) / col_std
    return df

def load_data(path):
    df = pd.read_csv(path)
    return df

def save_normalized(df, path):
    df.to_csv(path, index=False)

if __name__ == '__main__':
    train_df = load_data('data/part_ab_train.csv')
    # print(train_df.head())
    train_stats, normalized_train_df = normalize(train_df)
    val_df = load_data('data/part_ab_val.csv')
    test_df = load_data('data/part_ab_test_public.csv')
    normalized_val_df = fit_normalize(train_stats, val_df)
    normalized_test_df = fit_normalize(train_stats, test_df)

    save_normalized(normalized_train_df, path='data/normalized_train.csv')
    save_normalized(normalized_val_df, path='data/normalized_val.csv')
    save_normalized(normalized_test_df, path='data/normalized_test.csv')







