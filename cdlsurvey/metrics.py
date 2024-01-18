# third party
import numpy as np
import pandas as pd

# Pre-defined columns
LABEL_COLUMN = 'label'
PROTECTED_COLUMNS = ['gender_Female', 'gender_Male', 'race_White', 'race_Black']


def error_rate(labels, predictions):
    """
    Function to compute the error rate
    """
    signed_labels = (
        (labels > 0).astype(np.float32) - (labels <= 0).astype(np.float32),
    )

    # Assign the numerator
    numerator = (np.multiply(signed_labels.values, predictions.values) <= 0).sum()

    # Assign the denominator
    denominator = predictions.shape[0]

    return float(numerator) / float(denominator)


def tpr(df: pd.DataFrame) -> float:
    """
    Measure the true positive rate.
    """
    fp = sum((df['predictions'] >= 0.0) & (df[LABEL_COLUMN] > 0.5))
    ln = sum(df[LABEL_COLUMN] > 0.5)
    return float(fp) / float(ln)


def _get_error_rate_and_constraints(df, tpr_max_diff):
    """
    Computes the error and fairness violations.
    """
    error_rate_local = error_rate(df[['predictions']], df[[LABEL_COLUMN]])
    overall_tpr = tpr(df)
    return error_rate_local, [
        (overall_tpr - tpr_max_diff) - tpr(df[df[protected_attribute] > 0.5])
        for protected_attribute in PROTECTED_COLUMNS
    ]
