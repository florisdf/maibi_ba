import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


def get_X_from_df(df):
    return df[[*[f'V{i}' for i in range(1, 29)], 'scaled_amount', 'scaled_time']]


def get_Y_from_df(df):
    return df['Class']


def get_df(c_fp=2):
    """
    Args:
        c_fp (int, float): The (fixed) cost of a false positive prediction
    """
    df = pd.read_csv('data/creditcard_fraud_dataset.csv')
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df['C_FP'] = c_fp
    df['C_FN'] = df['Amount']

    one_hot_classes = np.zeros((len(df), 2))
    one_hot_classes[np.arange(len(df)), df['Class'].values] = 1

    costs = df[['C_FP', 'C_FN']].values
    df['C_misclf'] = np.multiply(costs, one_hot_classes).sum(axis=1)

    return df


def get_train_test_dfs(c_fp=2, test_size=0.5, random_state=42):
    """
    Args:
        c_fp (int, float): The (fixed) cost of a false positive prediction
        test_size (float): Relative size of the test set
        random_state (int): State for the random train-test split.
    """
    df = get_df(c_fp)
    X = get_X_from_df(df)
    Y = get_Y_from_df(df)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=random_state)

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    return df_train, df_test