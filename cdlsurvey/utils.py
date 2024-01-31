# stdlib
import random
from typing import Generator, List

# third party
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import torch
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error

# first party
from cdlsurvey.metrics import _get_error_rate_and_constraints


def training_generator(
    model, train_df, test_df, minibatch_size, num_iterations_per_loop=1, num_loops=1
) -> Generator:
    """
    Function that generates training data for tensorflow and tfco experimentation
    """
    # Set a random seed
    random.seed(123)

    # Get the size of the data to create minibatches
    num_rows = train_df.shape[0]

    # Minibatch size
    minibatch_size = min(minibatch_size, num_rows)

    # Shuffle the training data
    permutation = list(range(train_df.shape[0]))
    random.shuffle(permutation)

    # Start a tensorflow session
    session = tf.Session()
    session.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    # Define the indicies for the minibatch
    minibatch_start_index = 0
    for n in range(num_loops):
        for _ in range(num_iterations_per_loop):
            minibatch_indices: List[int] = []
            while len(minibatch_indices) < minibatch_size:
                minibatch_end_index = (
                    minibatch_start_index + minibatch_size - len(minibatch_indices)
                )
                if minibatch_end_index >= num_rows:
                    minibatch_indices += range(minibatch_start_index, num_rows)
                    minibatch_start_index = 0
                else:
                    minibatch_indices += range(
                        minibatch_start_index, minibatch_end_index
                    )
                    minibatch_start_index = minibatch_end_index

            # Run the tensorflow session
            session.run(
                model.train_op,
                feed_dict=model.feed_dict_helper(
                    train_df.iloc[[permutation[ii] for ii in minibatch_indices]]
                ),
            )

        # Get the training predictions
        train_predictions = session.run(
            model.predictions_tensor, feed_dict=model.feed_dict_helper(train_df)
        )

        # Get the test predictions
        test_predictions = session.run(
            model.predictions_tensor, feed_dict=model.feed_dict_helper(test_df)
        )

        yield (train_predictions, test_predictions)


def training_helper(
    model, train_df, test_df, minibatch_size, num_iterations_per_loop=1, num_loops=1
):
    """
    Function that will help with the training process for tfco
    """
    # Initialize lists for the results
    train_error_rate_vector = []
    train_constraints_matrix = []
    test_error_rate_vector = []
    test_constraints_matrix = []

    # Iterate over the training generator and calculate metrics
    for train, test in training_generator(
        model, train_df, test_df, minibatch_size, num_iterations_per_loop, num_loops
    ):
        # Assign predictions to dataframe
        train_df['predictions'] = train
        test_df['predictions'] = test

        # Compute the error rate and contraints for train
        train_error_rate, train_constraints = _get_error_rate_and_constraints(
            train_df, model.tpr_max_diff
        )
        train_error_rate_vector.append(train_error_rate)
        train_constraints_matrix.append(train_constraints)

        # Compute the error rate and constraints for test
        test_error_rate, test_constraints = _get_error_rate_and_constraints(
            test_df, model.tpr_max_diff
        )
        test_error_rate_vector.append(test_error_rate)
        test_constraints_matrix.append(test_constraints)

    return (
        train_error_rate_vector,
        train_constraints_matrix,
        test_error_rate_vector,
        test_constraints_matrix,
    )


def get_expgrad_models_per_epsilon(estimator, epsilon, X_train, y_train, A_train):
    """Instantiate and train an ExponentiatedGradient model on the
    balanced training dataset.

    Parameters
    ----------
    Estimator: Base estimator to contains a fit and predict function.
    Epsilon: Float representing maximum difference bound for the
    fairness Moment constraint

    Returns
    -------
    Predictors
        List of inner model predictors learned by the ExponentiatedGradient
        model during the training process.

    """
    exp_grad_est = ExponentiatedGradient(
        estimator=estimator,
        sample_weight_name='classifier__sample_weight',
        constraints=EqualizedOdds(difference_bound=epsilon),
    )
    # Is this an issue - Re-runs
    exp_grad_est.fit(X_train, y_train, sensitive_features=A_train)
    predictors = exp_grad_est.predictors_
    return predictors


def aggregate_predictor_performances(predictors, metric, X_test, Y_test, A_test=None):
    """Compute the specified metric for all classifiers in predictors.
    If no sensitive features are present, the metric is computed without
    disaggregation.

    Parameters
    ----------
    predictors: A set of classifiers to generate predictions from.
    metric: The metric (callable) to compute for each classifier in predictor
    X_test: The data features of the testing data set
    Y_test: The target labels of the teting data set
    A_test: The sensitive feature of the testing data set.

    Returns
    -------
    List of performance scores for each classifier in predictors, for the
    given metric.
    """
    all_predictions = [predictor.predict(X_test) for predictor in predictors]
    if A_test is not None:
        return [
            metric(Y_test, Y_sweep, sensitive_features=A_test)
            for Y_sweep in all_predictions
        ]
    else:
        return [metric(Y_test, Y_sweep) for Y_sweep in all_predictions]


def model_performance_sweep(models_dict, X_test, y_test, A_test):
    """Compute the equalized_odds_difference and balanced_error_rate for a
    given list of inner models learned by the ExponentiatedGradient algorithm.
    Return a DataFrame containing the epsilon level of the model, the index
    of the model, the equalized_odds_difference score and the balanced_error
    for the model.

    Parameters
    ----------
    models_dict: Dictionary mapping model ids to a model.
    X_test: The data features of the testing data set
    y_test: The target labels of the testing data set
    A_test: The sensitive feature of the testing data set.

    Returns
    -------
    DataFrame where each row represents a model (epsilon, index) and its
    performance metrics
    """
    performances = []
    for eps, models in models_dict.items():
        eq_odds_difference = aggregate_predictor_performances(
            models, equalized_odds_difference, X_test, y_test, A_test
        )
        bal_acc_score = aggregate_predictor_performances(
            models, balanced_accuracy_score, X_test, y_test
        )
        for i, score in enumerate(eq_odds_difference):
            performances.append((eps, i, score, (1 - bal_acc_score[i])))
    performances_df = pd.DataFrame.from_records(
        performances,
        columns=["epsilon", "index", "equalized_odds", "balanced_error"],
    )
    return performances_df


def calculate_starting_values(model, X_train, X_test, y_train, y_test, A_train, A_test):
    """
    Function to compute the values of the metrics before
    any training starts
    """
    # First we need to turn X_train and X_test into
    # torch.Tensors
    X_train = torch.tensor(X_train.values, dtype=torch.double)
    X_test = torch.tensor(X_test.values, dtype=torch.double)

    # Get the predictions
    x_train_predictions = model(X_train).flatten().detach().cpu().numpy()
    x_test_predictions = model(X_test).flatten().detach().cpu().numpy()

    # Masks
    train_male_mask = A_train == 1
    train_female_mask = A_train == 0
    test_male_mask = A_test == 1
    test_female_mask = A_test == 0

    # Calulate the differnece in MSE per group
    train_male_mse = mean_squared_error(
        y_train[train_male_mask], x_train_predictions[train_male_mask]
    )
    train_female_mse = mean_squared_error(
        y_train[train_female_mask], x_train_predictions[train_female_mask]
    )
    train_const = np.abs(train_male_mse - train_female_mse)

    test_male_mse = mean_squared_error(
        y_test[test_male_mask], x_test_predictions[test_male_mask]
    )
    test_female_mse = mean_squared_error(
        y_test[test_female_mask], x_test_predictions[test_female_mask]
    )
    test_const = np.abs(test_male_mse - test_female_mse)

    # Calculate the disparity in accuracy
    x_train_binary_preds = (x_train_predictions >= 0.5).astype(int)
    x_test_binary_preds = (x_test_predictions >= 0.5).astype(int)

    train_male_acc = accuracy_score(
        y_train[train_male_mask], x_train_binary_preds[train_male_mask]
    )
    train_female_acc = accuracy_score(
        y_train[train_female_mask], x_train_binary_preds[train_female_mask]
    )
    train_acc_disp = np.abs(train_male_acc - train_female_acc)

    test_male_acc = accuracy_score(
        y_test[test_male_mask], x_test_binary_preds[test_male_mask]
    )
    test_female_acc = accuracy_score(
        y_test[test_female_mask], x_test_binary_preds[test_female_mask]
    )
    test_acc_disp = np.abs(test_male_acc - test_female_acc)

    # Compute the overall MSE for train and test
    train_mse = mean_squared_error(y_train, x_train_predictions)
    test_mse = mean_squared_error(y_test, x_test_predictions)

    # Compute the overall accuracy
    train_acc = accuracy_score(y_train, x_train_binary_preds)
    test_acc = accuracy_score(y_test, x_test_binary_preds)

    return (
        [(train_const, test_const)],
        [(train_acc_disp, test_acc_disp)],
        [(train_mse, test_mse)],
        [(train_acc, test_acc)],
    )
