# stdlib
import random
from typing import Generator, List

# third party
import tensorflow.compat.v1 as tf


def training_generator(
    model, train_df, test_df, minibatch_size, num_iterations_per_loop=1, num_loops=1
) -> Generator:
    """
    Function that generates training data for tensorflow and tfco experimentation
    """
    # Set a random seed
    random.seed(31337)

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

            session.run(
                model.train_op,
                feed_dict=model.feed_dict_helper(
                    train_df.iloc[[permutation[ii] for ii in minibatch_indices]]
                ),
            )

        train_predictions = session.run(
            model.predictions_tensor, feed_dict=model.feed_dict_helper(train_df)
        )
        test_predictions = session.run(
            model.predictions_tensor, feed_dict=model.feed_dict_helper(test_df)
        )

        yield (train_predictions, test_predictions)
