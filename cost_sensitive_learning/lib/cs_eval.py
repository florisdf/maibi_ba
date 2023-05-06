def get_tp_fp_fn_amount(y_true, y_pred, amounts, fp_cost=2.0):
    """
    Return the monetary value of the true positives, false positives
    and false negatives.
    """
    total_fraud_amount = (amounts * y_true).sum()
    tp_amount = (amounts * y_true * y_pred).sum()
    fn_amount = total_fraud_amount - tp_amount

    tps = (y_true * y_pred).sum()
    fps = y_pred.sum() - tps
    fp_amount = fp_cost*fps

    return tp_amount, fp_amount, fn_amount


def evaluate_pred(y_true, y_pred, amounts, fp_cost=2.0):
    tp_amount, fp_amount, fn_amount = get_tp_fp_fn_amount(y_true, y_pred, amounts, fp_cost=fp_cost)

    return {
        'Cost Precision': tp_amount/(tp_amount + fp_amount),
        'Cost Recall': tp_amount/(tp_amount + fn_amount),
        'TP Amount': tp_amount,
        'FP Amount': fp_amount,
        'FN Amount': fn_amount,
        'Net Recovered Amount': tp_amount - fp_amount - fn_amount,
    }