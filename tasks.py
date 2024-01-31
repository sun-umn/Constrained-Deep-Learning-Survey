# stdlib
import os

# third party
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from skorch.callbacks import EpochScoring
from skorch.dataset import Dataset
from skorch.helper import predefined_split

# first party
from cdl_python.core.models import FairlearnMLP
from cdlsurvey.cdls import FairClassifier
from cdlsurvey.data import get_wenjie_data_for_fairlearn
from cdlsurvey.metrics import accuracy_disparity, feasibility
from cdlsurvey.utils import get_expgrad_models_per_epsilon, model_performance_sweep

# Main directory
MAIN_DIR = '/home/jusun/dever120/Constrained-Deep-Learning-Survey'


def run_pytorch_fairlearn(epsilon: float) -> None:
    """
    Function that will run pytorch MLP model
    with fairlearn for constrained deep learning.
    """
    # Load in the adult dataset created by Wenjie
    X_train, X_test = get_wenjie_data_for_fairlearn()

    # Set up the variables for the reductions
    y_train = X_train['income'].values
    y_test = X_test['income'].values
    A_train = X_train['sex'].values
    A_test = X_test['sex'].values

    # Remove values from training and test
    X_train = X_train.drop(columns=['sex', 'income'])
    X_test = X_test.drop(columns=['sex', 'income'])

    # Create the model
    mlp_model = FairlearnMLP(
        num_features=X_train.shape[1],
        num_classes=1,
        training_samples=len(X_train),
        testing_samples=len(X_test),
        sensitive_group={'train': A_train, 'test': A_test},
    )

    # For skorch since we already have test data we
    # need to input the test data as a skorch dataset
    test_ds = Dataset(X_test.values, y_test)

    # Build the callbacks for the feasibility measure - for these
    # measures we need to build one for train AND test due to the
    # on_train parameter
    train_feasibility = EpochScoring(
        scoring=feasibility,
        lower_is_better=True,
        on_train=True,
        name='train_feasibility',
    )
    test_feasibility = EpochScoring(
        scoring=feasibility,
        lower_is_better=True,
        on_train=False,
        name='test_feasibility',
    )

    # Build the callbacks for the accuracy disparity measure
    train_accuracy_disparity = EpochScoring(
        scoring=accuracy_disparity,
        lower_is_better=True,
        on_train=True,
        name='train_accuracy_disparity',
    )
    test_accuracy_disparity = EpochScoring(
        scoring=accuracy_disparity,
        lower_is_better=True,
        on_train=False,
        name='test_accuracy_disparity',
    )

    # Callback for accuracy
    accuracy = EpochScoring(
        scoring='accuracy', lower_is_better=False, on_train=True, name='train_accuracy'
    )

    # Initialize the torch model for skoch and fairlearn
    # Remember we need reductions='none' here to mimic a sample_weight
    # parameter in the get_loss function
    torch_model = FairClassifier(
        mlp_model,
        criterion=torch.nn.MSELoss(reduction='none'),
        optimizer=torch.optim.Adam,
        train_split=predefined_split(test_ds),
        lr=0.0001,
        max_epochs=300,
        batch_size=len(X_train),
        callbacks=[
            accuracy,
            train_feasibility,
            test_feasibility,
            train_accuracy_disparity,
            test_accuracy_disparity,
        ],
    )

    # MLP from our cdl survey
    estimator = Pipeline(
        steps=[
            ("classifier", torch_model),
        ]
    )

    # Define the epsilons
    epsilons = [epsilon]

    # NOTE: Training for all models with this epsilon / r in our paper
    all_models = {}
    for eps in epsilons:
        all_models[eps] = get_expgrad_models_per_epsilon(
            estimator=estimator,
            epsilon=eps,
            X_train=X_train,
            y_train=y_train,
            A_train=A_train,
        )

    # For logging purposes during training
    # The method finds different models based on a sample
    # weight of the data for each epsilon
    for epsilon, models in all_models.items():
        print(
            f'For epsilon {epsilon}, ExponentiatedGradient'
            f'learned {len(models)} inner models'
        )

    # Get the model with the best performance and create our metrics
    # from that model
    performance_df = model_performance_sweep(all_models, X_test, y_test, A_test)
    best_model_by_eq_oddds = performance_df.sort_values('equalized_odds').reset_index(
        drop=True
    )
    epsilon, index = (
        best_model_by_eq_oddds.iloc[0, :]['epsilon'],
        best_model_by_eq_oddds.iloc[0, :]['index'],
    )
    inprocess_model = all_models[epsilon][index]

    # We care about the metrics we defined
    # Disparity between the loss of the groups
    # Disparity between the accuracy of the groups
    # Objective loss in general
    # Accuracy in general
    # Get the feasibility / constraint
    feasibility_df = pd.DataFrame(
        [
            (i['train_feasibility'], i['test_feasibility'])
            for i in inprocess_model.steps[1][1].history
        ],
        columns=['train', 'test'],
    )

    # Get the accuracy disparity
    acc_disp_df = pd.DataFrame(
        [
            (i['train_accuracy_disparity'], i['test_accuracy_disparity'])
            for i in inprocess_model.steps[1][1].history
        ],
        columns=['train', 'test'],
    )

    # Get the objective trajectory
    loss_df = pd.DataFrame(
        [
            (i['train_loss'], i['valid_loss'])
            for i in inprocess_model.steps[1][1].history
        ],
        columns=['train', 'test'],
    )

    # Get the accuracy
    acc_df = pd.DataFrame(
        [
            (i['train_accuracy'], i['valid_acc'])
            for i in inprocess_model.steps[1][1].history
        ],
        columns=['train', 'test'],
    )

    # Set up dictionary and save the data
    data_dict = {
        'diff': feasibility_df.values,
        'f': loss_df.values,
        'accuracy_diff': acc_disp_df.values,
        'accuracy': acc_df.values,
    }

    # Save filepath
    filename = os.path.join(MAIN_DIR, 'data', 'fairlearn_results.npz')
    np.savez(filename, **data_dict)
