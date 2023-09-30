import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def remove_outliers(df: pd.DataFrame, outlier_percent: float = 0.02) -> pd.DataFrame:
    """
    Remove outliers from the dataframe based on the provided outlier percent.

    Parameters:
    - df (pd.DataFrame): The dataframe from which outliers need to be removed.
    - outlier_percent (float, default=0.02): The percentage of outliers to remove
      from the top and bottom of the data distribution.

    Returns:
    - pd.DataFrame: DataFrame after removing outliers.

    Example:
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100], "B": ["a", "b", "c", "d", "e", "f"]})
    >>> df_no_outliers = remove_outliers(df, outlier_percent=0.2)
    """

    for column in df.select_dtypes(include=[np.number]).columns:
        lower_threshold = df[column].quantile(outlier_percent)
        upper_threshold = df[column].quantile(1 - outlier_percent)
        df = df[(df[column] >= lower_threshold) & (df[column] <= upper_threshold)]

    return df


def drop_low_variance_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop columns with low variance.

    Parameters:
    - df (pd.DataFrame): DataFrame to be processed.
    - threshold (float, optional): Variance threshold, default is 0.9.

    Returns:
    - pd.DataFrame: DataFrame after dropping low variance columns.

    Example:
    >>> df = pd.DataFrame({"A": [1,1,1,1,2], "B": [1,2,3,4,5]})
    >>> updated_df = drop_low_variance_columns(df, 0.8)
    """

    freq = df.apply(lambda x: x.value_counts().max()) / len(df)
    df = df.drop(columns=freq[freq >= threshold].index.tolist())
    return df


def drop_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns in the DataFrame that are not of numeric type (int or float).

    Parameters:
    - df (pd.DataFrame): The DataFrame from which non-numeric columns need to be dropped.

    Returns:
    - pd.DataFrame: DataFrame containing only numeric columns.

    Example:
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
    >>> updated_df = drop_non_numeric_columns(df)
    >>> print(updated_df)
       A    C
    0  1  1.1
    1  2  2.2
    2  3  3.3
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return df[numeric_cols]


def draw_correlations(
    df: pd.DataFrame,
    target_col: str,
    methods: list = ["pearson", "spearman", "kendall"],
    num_top_features: int = 10,
):
    """
    Draw correlation heatmaps between numeric columns and a target column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The name of the target column for correlation analysis.
        methods (list, optional): List of correlation methods to use. Default is ['pearson', 'spearman', 'kendall'].
        num_top_features (int, optional): Number of top correlated features to display. Default is 10.

    Returns:
        None
    """
    df_numeric = df.select_dtypes(include=["number"])

    # Columns to drop (those with a single dominant value)
    drop_columns = []
    for col in df_numeric.columns:
        most_frequent = df_numeric[col].value_counts().idxmax()
        if (df_numeric[col] == most_frequent).mean() >= 0.9:
            drop_columns.append(col)

    # Remove columns with dominant values
    df_filtered = df_numeric.drop(columns=drop_columns)

    # Calculate correlations with 'target_col'
    correlations_target = {}
    for method in methods:
        corr_series = df_filtered.corrwith(df_filtered[target_col], method=method)
        top_correlations = corr_series.abs().nlargest(num_top_features)
        correlations_target[method] = top_correlations

    # Draw correlation heatmaps for 'target_col'
    for method, top_correlations in correlations_target.items():
        plt.figure(figsize=(10, 10))
        corr_matrix = df_filtered[top_correlations.index].corr(method=method)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=False)
        plt.title(
            f"{method.capitalize()} Correlation with {target_col} (Top {num_top_features})"
        )
        plt.show()
