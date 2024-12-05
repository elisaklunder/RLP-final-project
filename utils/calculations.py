import numpy as np
import pandas as pd
from tabulate import tabulate


def calculate_mean_and_se(
    results: pd.DataFrame, value_columns: list, hyperparameter_columns: list
):
    """
    Calculate the mean and standard error (SE) for specified value columns,
    grouped by hyperparameters only.

    Args:
    - results: DataFrame containing the dataset.
    - value_columns: List of columns to calculate mean and SE for (e.g., `['mean_ep_rew', 'mean_ep_len']`).
    - hyperparameter_columns: List of hyperparameter columns to group by.

    Returns:
    - aggregated_results: DataFrame with mean and SE for specified value columns grouped by hyperparameters.
    """
    grouped = results.groupby(hyperparameter_columns)

    agg_dict = {}
    for value_col in value_columns:
        agg_dict[f"mean_{value_col}"] = (value_col, "mean")
        agg_dict[f"sem_{value_col}"] = (
            value_col,
            lambda x: np.sqrt((x**2).sum()) / len(x),
        )

    aggregated_results = grouped.agg(**agg_dict).reset_index()
    print(tabulate(aggregated_results, headers="keys", tablefmt="grid"))
