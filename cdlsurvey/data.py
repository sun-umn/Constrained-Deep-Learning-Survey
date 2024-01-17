# stdlib
from typing import List, Tuple, Union

# third party
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


def binarize_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to binarize columns for train and test there may
    be categories in test that we do not see in train and vice versa.

    TODO: I think it makes more sense to just binarize data that is in train because
    the signal will be zero for those training samples.
    """
    # Get the categorical columns
    categorical_train_df = train_df[columns]
    categorical_test_df = test_df[columns]

    # Concatenate the train and test data
    categorical_train_df['is_train'] = 1
    categorical_test_df['is_train'] = 0

    df = pd.concat([categorical_train_df, categorical_test_df], axis=0)

    # Binarize the columns with pandas get dummies
    df = pd.get_dummies(df).astype(int)

    # Split the data back into train and test
    categorical_train_df = df.query('is_train == 1').reset_index()
    categorical_test_df = df.query('is_train == 0').reset_index()

    return categorical_train_df, categorical_test_df


def discretize_continuous_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column_name: str,
    *,
    num_quantiles: Union[int, None] = None,
    bins: Union[List[int], None] = None
) -> None:
    """
    Function to discretize continuous columns in a dataset
    """
    pass


def get_data():
    """
    Function to build the Adult dataset for binary classification
    with fairness-constraints
    """
    train_filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'  # noqa
    test_filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'  # noqa

    # Get the train and test data
    train_df = pd.read_csv(train_filename, names=COLUMNS, skipinitialspace=True)
    test_df = pd.read_csv(test_filename, names=COLUMNS, skipinitialspace=True)

    # Let's add an assertion that all of the columns are the same
    assert train_df.column == test_df.columns

    return train_df, test_df
