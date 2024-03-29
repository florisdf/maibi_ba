import pandas as pd
from sklearn.linear_model import LogisticRegression

from .creditcard_fraud_dataset import X_COLS, Y_COL


def train_clf(
    df_train,
    n_pos=-1, n_neg=-1,
    class_weight=None,
    sample_weight=None,
    max_iter=500,
    random_state=42,
    Classifier=LogisticRegression,
):
    """
    Args:
        df_train (pd.DataFrame): DataFrame containing the training samples and classes.
        n_pos (int): the number of positives to randomly sample. If -1,
            all positives will be used.
        n_neg (int): the number of negatives to randomly sample. If -1,
            all negatives will be used.
        class_weight (str): passed as the `class_weight` arg of the model constructor.
        sample_weight (pd.Series): The weight of each corresponding training sample.
            If `None`, all samples will receive equal weight.
    """
    X_train = df_train[X_COLS]
    Y_train = df_train[Y_COL]

    # Create binary masks to select positives and negatives
    neg_mask = Y_train == 0
    pos_mask = Y_train == 1

    if n_pos == -1:
        # Use all positives
        n_pos = pos_mask.sum()
    if n_neg == -1:
        # Use all negatives
        n_neg = neg_mask.sum()

    # Choose subset of positives and negatives to train on
    neg_idxs_sub = X_train[neg_mask].sample(n=n_neg,
                                            random_state=random_state).index
    pos_idxs_sub = X_train[pos_mask].sample(n=n_pos,
                                            random_state=random_state).index

    # Use indices to select positives and negatives from X_train
    X_train_sub = pd.concat([X_train.loc[neg_idxs_sub],
                             X_train.loc[pos_idxs_sub]])
    Y_train_sub = pd.concat([Y_train.loc[neg_idxs_sub],
                             Y_train.loc[pos_idxs_sub]])


    if sample_weight is not None:
        sample_weight = pd.concat([
            sample_weight.loc[neg_idxs_sub],
            sample_weight.loc[pos_idxs_sub],
        ]).values
    df_train_sub = pd.concat([df_train.loc[neg_idxs_sub],
                              df_train.loc[pos_idxs_sub]])

    # TODO: improve this
    if Classifier == LogisticRegression:
        clf = Classifier(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        clf = Classifier(
            random_state=random_state,
        )
    return clf.fit(
        X_train_sub,
        Y_train_sub,
        sample_weight=sample_weight
    )