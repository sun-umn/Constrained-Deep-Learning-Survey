# stdlib
from typing import List, Tuple, Union

# third party
import numpy as np
import pandas as pd

# Define the columns that we want to use
# Categorical columns
CATEGORICAL_COLUMNS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country',
]

# Continuous columns
CONTINUOUS_COLUMNS = [
    'age',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'education_num',
]

# All columns
COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket',
]

# label column
LABEL_COLUMN = 'label'


def binarize_categorical_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to binarize columns for train and test there may
    be categories in test that we do not see in train and vice versa.

    TODO: I think it makes more sense to just binarize data that is in train because
    the signal will be zero for those training samples.

    TODO: Why does my dataset create an error?
    """
    # Get the categorical columns
    categorical_train_df = train_df[columns]
    categorical_test_df = test_df[columns]

    # Concatenate the train and test data
    categorical_train_df['is_train'] = 1
    categorical_test_df['is_train'] = 0

    df = pd.concat([categorical_train_df, categorical_test_df], axis=0)

    # Binarize the columns with pandas get dummies
    df = pd.get_dummies(df)

    # Split the data back into train and test
    categorical_train_df = df.query('is_train == 1').reset_index(drop=True)
    categorical_test_df = df.query('is_train == 0').reset_index(drop=True)

    # Remove is train data
    categorical_train_df = categorical_train_df.drop(columns=['is_train'])
    categorical_test_df = categorical_test_df.drop(columns=['is_train'])

    return categorical_train_df, categorical_test_df


def discretize_continuous_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column_name: str,
    *,
    num_quantiles: Union[int, None] = None,  # noqa
    bins: Union[List[int], None] = None  # noqa
) -> None:
    """
    Function to discretize continuous columns in a dataset. Functions will
    modify the data within the function
    """
    assert (
        num_quantiles is None or bins is None
    ), "num quantiles and bins cannot both be None for this function"

    # quantile featurizing
    if num_quantiles is not None:
        _, bins_quantized = pd.qcut(
            train_df[column_name],
            num_quantiles,
            retbins=True,  # Whether to return the (bins, labels) or not. Can be useful if bins is given as a scalar.  # noqa
            labels=False,
        )

        # Add feature to train
        train_df[column_name] = pd.cut(
            train_df[column_name],
            bins_quantized,
            labels=False,
            include_lowest=True,
        )

        # Add feature to test
        test_df[column_name] = pd.cut(
            test_df[column_name], bins_quantized, labels=False, include_lowest=True
        )

    elif bins is not None:
        # Add feature to train
        train_df[column_name] = pd.cut(
            train_df[column_name],
            bins,
            labels=False,
            include_lowest=True,
        )

        # Add feature to test
        test_df[column_name] = pd.cut(
            test_df[column_name], bins, labels=False, include_lowest=True
        )


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Function to build the Adult dataset for binary classification
    with fairness-constraints
    """
    USEABLE_COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + [LABEL_COLUMN]

    train_filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'  # noqa
    test_filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'  # noqa

    # Get the train and test data
    train_df = pd.read_csv(train_filename, names=COLUMNS, skipinitialspace=True)
    test_df = pd.read_csv(
        test_filename, names=COLUMNS, skipinitialspace=True, skiprows=1
    )

    # Create the label column - the label is for the income bracket column
    # and we identify anyone who makes over $50k
    train_df[LABEL_COLUMN] = (
        train_df['income_bracket'].apply(lambda x: x == '>50K')
    ).astype(int)
    test_df[LABEL_COLUMN] = (
        test_df['income_bracket'].apply(lambda x: x == '>50K')
    ).astype(int)

    # Let's add an assertion that all of the columns are the same
    assert np.all(train_df.columns == test_df.columns)

    # Default = warn
    pd.options.mode.chained_assignment = None

    # Process & featurize the data
    # First filter out irrelevant columns
    train_df = train_df[USEABLE_COLUMNS].copy()
    test_df = test_df[USEABLE_COLUMNS].copy()

    # Discretize columns
    discretize_continuous_columns(train_df, test_df, 'age', num_quantiles=4)
    discretize_continuous_columns(
        train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000]
    )
    discretize_continuous_columns(
        train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500]
    )
    discretize_continuous_columns(
        train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100]
    )
    discretize_continuous_columns(
        train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16]
    )

    # Binarize the columns
    train_df, test_df = binarize_categorical_columns(
        train_df, test_df, columns=USEABLE_COLUMNS
    )

    # Get feature names
    feature_names = train_df.columns.tolist()
    feature_names.remove(LABEL_COLUMN)

    return train_df, test_df, feature_names
