import numpy as np
from sklearn.metrics import (
    precision_score, recall_score,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
)

from sklearn.metrics._ranking import _binary_clf_curve

from .creditcard_fraud_dataset import get_X_from_df


def get_tp_fp_fn_amount(df_test, y_pred):
    """
    Return the monetary value of the true positives, false positives
    and false negatives.
    """
    assert len(df_test) == len(y_pred)
    assert len(df_test['C_FP'].unique()) == 1

    y_true = df_test['Class']
    amounts = df_test['Amount']

    total_fraud_amount = (amounts * y_true).sum()
    tp_amount = (amounts * y_true * y_pred).sum()
    fn_amount = total_fraud_amount - tp_amount

    tps = (y_true * y_pred).sum()
    fps = y_pred.sum() - tps
    fp_amount = df_test['C_FP'].iloc[0]*fps

    return tp_amount, fp_amount, fn_amount


def get_threshold_idxs(y_score):
    """
    Return the indexes used by sklearn to obtain the thresholds in _binary_clf_curve()
    
    See https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/metrics/_ranking.py#L694
    """
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    distinct_value_indices = np.where(np.diff(y_score[desc_score_indices]))[0]
    return desc_score_indices[
        np.r_[distinct_value_indices, y_score.size - 1]
    ]


def evaluate_clf(clf, df_test, thresh=0.5):
    X_test = get_X_from_df(df_test)

    y_proba = clf.predict_proba(X_test)
    y_pred = np.zeros(len(y_proba))
    y_pred[y_proba[:, 1] > thresh] = 1

    tp_amount, fp_amount, fn_amount = get_tp_fp_fn_amount(df_test, y_pred)

    y_true = df_test['Class']
    y_score = clf.decision_function(X_test)

    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'ap': average_precision_score(y_true, y_score),
        'pr_curve': precision_recall_curve(y_true, y_score),
        'tp_amount': tp_amount,
        'fp_amount': fp_amount,
        'fn_amount': fn_amount,
    }


def get_cost_sensitive_pr_curve(df_test, y_score):
    raise NotImplementedError

    y_true = df_test['Class']

    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score
    )
    fns = tps[-1] - tps

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    df_sorted = df_test.iloc[desc_score_indices]
    df_pos = df_sorted[df_sorted['Class'] == 1]
    c_fn = df_pos['C_FN'].values

    fns_uniq, fns_uniq_inv_idxs = np.unique(fns, return_inverse=True)

    # Since c_fn is reversely sorted wrt y_score, any N false negatives
    # correspond to the N last costs in c_fn. The cost that corresponds to
    # N false negatives, therefore, is equal to sum(c_fn[-N:]), which
    # is the same as sum(c_fn[::-1][:N]).
    #
    # Element a.cumsum()[i] of the cumulative sum array of an array a, is defined
    # as a.cumsum()[i] == sum(a[:i]). Hence,
    #
    # c_fn[::-1].cumsum()[N] == sum(c_fn[::-1][:N]),
    #
    # which is the same as sum(c_fn[-N:]). The false negatives in c_fn, however,
    # are sorted from highest to lowest number of false negatives. Therefore,
    # to obtain the cost of fn[i] false negatives, we need the element at
    # c_fn[::-1].cumsum()[-i], which is the same as c_fn[::-1].cumsum()[::-1][i].
    #
    # Remember that np.unique() returns sorted unique array of fns, but c_fn is reversely sorted
    # wrt the number of fns. We want indices for c_fn, so we need to reverse 
    # the indices returned by np.unique().
    fn_amounts = c_fn[::-1].cumsum()[fns_uniq_inv_idxs]

    # TODO: Finish this function