# stdlib
from typing import List, Union

# third party
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco
import torch
from skorch import NeuralNetBinaryClassifier
from tensorflow.python.framework import ops


class FairClassifier(NeuralNetBinaryClassifier):
    """
    Customized skorch wrapper build on top of
    NeuralNetBinaryClassifier to be utilized with
    the fairlearn library
    """

    def predict(self, X):
        predictions = self.predict_proba(X)
        predictions = (predictions > self.threshold).astype(int)
        return predictions

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = torch.tensor(y_true, dtype=torch.double, device=self.device)
        # Train condition
        train_sample_weight = self.module_.train_sample_weight
        test_sample_weight = self.module_.test_sample_weight
        train_condition = (
            train_sample_weight is not None
            and self.module_.training_samples == len(y_true)
        )
        test_condition = (
            test_sample_weight is not None
            and self.module_.testing_samples == len(y_true)
        )
        if train_condition:
            sample_weight = self.module_.train_sample_weight
        elif test_condition:
            sample_weight = self.module_.test_sample_weight
        else:
            sample_weight = None

        if sample_weight is not None:
            loss = self.criterion(y_pred, y_true)
            loss = loss * torch.tensor(sample_weight, dtype=torch.double)
            loss = loss.mean()
        else:
            loss = self.criterion_(y_pred, y_true).mean()

        return loss


class Model:
    """
    Model for cdl-survey experiments. This defined a tensorflow model
    for fairness and constrained optimization.
    """

    def __init__(
        self,
        tpr_max_diff: float,
        protected_columns: List[str],
        feature_names: Union[List[str]] = ['label'],
        label_column: str = 'label',
    ) -> None:
        # Set a tensorflow random seed - came from notebook
        tf.random.set_random_seed(123)

        # Set true positive rate max difference
        self.tpr_max_diff = tpr_max_diff

        # Set the feature names
        self.feature_names = feature_names

        # Set protected columns
        self.protected_columns = protected_columns

        # Label column
        self.label_column = label_column

        # Get the number of features
        num_features = len(self.feature_names)

        # Setup placeholder columns
        # feature placeholder
        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(None, num_features), name='features_placeholder'
        )

        # Labels placeholder
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder'
        )

        # Protected features placeholder
        if self.protected_columns is None:
            raise ValueError('Protected columns needs to be defined!')

        self.protected_placeholders = [
            tf.placeholder(tf.float32, shape=(None, 1), name=attribute + "_placeholder")
            for attribute in self.protected_columns
        ]

        # Set up a 2 layer neural network
        self.first_hidden_layer = tf.layers.dense(
            inputs=self.features_placeholder,
            units=64,
            activation=tf.nn.relu,
        )

        # Add bn after first layer
        self.bn1 = tf.layers.batch_normalization(self.first_hidden_layer)

        # Second hidden layer
        self.second_hidden_layer = tf.layers.dense(
            inputs=self.bn1,
            units=64,
            activation=tf.nn.relu,
        )

        # Add bn after the second layer
        self.bn2 = tf.layers.batch_normalization(self.second_hidden_layer)

        # We use a linear model
        # This study uses a very simple linear model
        self.predictions_tensor = tf.layers.dense(
            inputs=self.bn2, units=1, activation=None
        )

    def build_train_op(
        self, learning_rate: float, *, unconstrained: bool = False
    ) -> ops.Operation:
        """
        Training operation for tensorflow and constrained optimization
        """
        # set the tensorflow context
        ctx = tfco.rate_context(self.predictions_tensor, self.labels_placeholder)

        # Get a positive slice
        positive_slice = ctx.subset(self.labels_placeholder > 0)

        # Overall true positive rate
        overall_tpr = tfco.positive_prediction_rate(positive_slice)

        # Set a list of constraints
        constraints = []
        if not unconstrained:
            for placeholder in self.protected_placeholders:
                slice_tpr = tfco.positive_prediction_rate(
                    ctx.subset((placeholder > 0) & (self.labels_placeholder > 0))
                )
                constraints.append(
                    slice_tpr >= overall_tpr - self.tpr_max_diff
                )  # This is where we set the constraint

        # Minimization problem
        mp = tfco.RateMinimizationProblem(tfco.error_rate(ctx), constraints)

        # Optimization setup
        opt = tfco.ProxyLagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))

        # Set the training op
        self.train_op = opt.minimize(mp)

        return self.train_op

    def feed_dict_helper(self, dataframe: pd.DataFrame) -> dict:
        """
        Function to feed dictionary
        """
        feed_dict = {
            self.features_placeholder: dataframe[self.feature_names],
            self.labels_placeholder: dataframe[[self.label_column]],
        }

        # Iterate over dict
        for i, protected_attribute in enumerate(self.protected_columns):
            feed_dict[self.protected_placeholders[i]] = dataframe[[protected_attribute]]

        return feed_dict