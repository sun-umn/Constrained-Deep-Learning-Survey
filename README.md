![](https://github.com/sun-umn/Constrained-Deep-Learning-Survey/blob/main/cdl-survey.png)

# Constrained-Deep-Learning-Survey
Discover algorithms for deep learning under constraints.

## Get Data

The data that we are currently (2024-01-23) using is the [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) and the function to build the data can be found under `cdlsurvey.data`. It is a general function for building the Adult dataset to explore constrained deep learning. To use the function:

```python
from cdlsurvey.data import get_data

train_df, test_df, feature_names = get_data()
```