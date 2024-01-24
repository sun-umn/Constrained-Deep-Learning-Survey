# stdlib
import random
from typing import Generator, List

# third party
import tensorflow.compat.v1 as tf

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
