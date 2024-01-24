# third party
import numpy as np
import pandas as pd

# Pre-defined columns
LABEL_COLUMN = 'label'
PROTECTED_COLUMNS = ['gender_Female', 'gender_Male', 'race_White', 'race_Black']


def error_rate(predictions, labels):
    """
    Function that computes the error rate
    """
    signed_labels = (labels > 0).astype(np.float32) - (labels <= 0).astype(np.float32)
    numerator = (np.multiply(signed_labels.values, predictions.values) <= 0).sum()
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
    Function that computes the error and fairness violations.
    """
    error_rate_local = error_rate(df[['predictions']], df[[LABEL_COLUMN]])
    overall_tpr = tpr(df)
    return error_rate_local, [
        (overall_tpr - tpr_max_diff) - tpr(df[df[protected_attribute] > 0.5])
        for protected_attribute in PROTECTED_COLUMNS
    ]


def get_exp_error_rate_constraints(cand_dist, error_rates_vector, constraints_matrix):
    """
    Function that computes the expected error and
    fairness violations on a randomized solution.
    """
    expected_error_rate = np.dot(cand_dist, error_rates_vector)
    expected_constraints = np.matmul(cand_dist, constraints_matrix)
    return expected_error_rate, expected_constraints
