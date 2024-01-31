# third party
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

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


def compute_error_metric(metric_value, sample_size):
    """Compute standard error of a given metric based on the assumption of
    normal distribution.

    Parameters:
    metric_value: Value of the metric
    sample_size: Number of data points associated with the metric

    Returns:
    The standard error of the metric
    """
    metric_value = metric_value / sample_size
    return 1.96 * np.sqrt(metric_value * (1.0 - metric_value)) / np.sqrt(sample_size)


def false_positive_error(y_true, y_pred):
    """Compute the standard error for the false positive rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fp, tn + fp)


def false_negative_error(y_true, y_pred):
    """Compute the standard error for the false negative rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fn, fn + tp)


def balanced_accuracy_error(y_true, y_pred):
    """Compute the standard error for the balanced accuracy estimate."""
    fpr_error, fnr_error = false_positive_error(y_true, y_pred), false_negative_error(
        y_true, y_pred
    )
    return np.sqrt(fnr_error**2 + fpr_error**2) / 2


def feasibility(model, X, y):
    """
    Check the feasibility - the differences of the loss
    and the differences of the accuracy
    """
    train_group = model.module_.sensitive_group['train']
    test_group = model.module_.sensitive_group['test']

    inputs = torch.tensor(X.X, dtype=torch.double)
    predictions = model.module_(inputs).detach().cpu().numpy()

    if len(y) == len(train_group):
        male_mask = train_group == 1
        female_mask = train_group == 0

        # Get the mse for each group
        male_mse = mean_squared_error(y[male_mask], predictions[male_mask])
        female_mse = mean_squared_error(y[female_mask], predictions[female_mask])

        return np.abs(male_mse - female_mse)

    else:
        male_mask = test_group == 1
        female_mask = test_group == 0

        # Get the mse for each group
        male_mse = mean_squared_error(y[male_mask], predictions[male_mask])
        female_mse = mean_squared_error(y[female_mask], predictions[female_mask])

        return np.abs(male_mse - female_mse)


# Create a new call back to look at feasibility
def accuracy_disparity(model, X, y):
    """
    Check the feasibility - the differences of the loss
    and the differences of the accuracy
    """
    train_group = model.module_.sensitive_group['train']
    test_group = model.module_.sensitive_group['test']

    inputs = torch.tensor(X.X, dtype=torch.double)
    predictions = (model.module_(inputs).detach().cpu().numpy() > 0.5).astype(int)

    if len(y) == len(train_group):
        male_mask = train_group == 1
        female_mask = train_group == 0

        # Get the mse for each group
        male_acc = accuracy_score(y[male_mask], predictions[male_mask])
        female_acc = accuracy_score(y[female_mask], predictions[female_mask])

        return np.abs(male_acc - female_acc)

    else:
        male_mask = test_group == 1
        female_mask = test_group == 0

        # Get the mse for each group
        male_acc = accuracy_score(y[male_mask], predictions[male_mask])
        female_acc = accuracy_score(y[female_mask], predictions[female_mask])

        return np.abs(male_acc - female_acc)
