import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# The feature columns
X_COLS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
          'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
          'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
          'Scaled Amount', 'Scaled Time']
Y_COL = 'Class'


def get_df(c_fp=2):
    """
    Args:
        c_fp (int, float): The (fixed) cost of a false positive prediction
    """
    df = pd.read_csv('data/creditcard_fraud_dataset.csv')
    rob_scaler = RobustScaler()

    df['Scaled Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['Scaled Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

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
    X = df[X_COLS]
    y = df[Y_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    return df_train, df_test