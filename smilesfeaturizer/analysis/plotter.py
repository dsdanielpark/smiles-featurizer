import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_corr(
    df: pd.DataFrame,
    target_col: str,
    methods: list = ["pearson", "spearman", "kendall"],
    num_top_features: int = 10,
):
    """
    Create correlation plots for specified correlation methods and visualize the top correlated features.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing both features and the target column.
        target_col (str): The name of the target column.
        methods (list, optional): List of correlation methods to compute ('pearson', 'spearman', 'kendall'). Default is ['pearson', 'spearman', 'kendall'].
        num_top_features (int, optional): Number of top correlated features to visualize. Default is 10.

    Example:
        # Example usage
        >>> data = {
        ...     'Feature1': [1, 2, 3, 4, 5],
        ...     'Feature2': [5, 4, 3, 2, 1],
        ...     'Feature3': [2, 2, 2, 2, 2],
        ...     'Target': [10, 20, 30, 40, 50]
        ... }
        >>> df = pd.DataFrame(data)
        >>> draw_corr(df, target_col='Target', methods=['pearson', 'spearman'], num_top_features=2)
        # This will display correlation heatmaps for 'Target' with 'Feature1' and 'Feature2' using Pearson and Spearman methods.
    """
    # Select only numeric columns and remove columns with dtype 'object'

    # Columns to drop (those with a single dominant value)
    drop_columns = []
    df_numeric = df.select_dtypes(include=["number"])
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
        plt.figure(figsize=(8, 8))
        corr_matrix = df_filtered[top_correlations.index].corr(method=method)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=False)
        plt.title(
            f"{method.capitalize()} Correlation with {target_col} (Top {num_top_features})"
        )
        plt.show()


def df_scatter_plot(
    df: pd.DataFrame, folder_path: str = None, smiles_col: str = "SMILES"
):
    """
    Create scatter plots for each column in the DataFrame against the index.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        folder_path (str, optional): The path to the folder where images will be saved. Default is None.

    Example:
        # Example usage:
        # df_scatter_plot(df, folder_path="scatter_plots_folder", smiles_col="SMILES")
    """
    # Create the folder if it doesn't exist
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
    df_numeric = df.select_dtypes(include=["number"])
    for column in df_numeric.columns:
        if column in [smiles_col, "id"]:
            pass
        else:
            try:
                plt.figure(figsize=(8, 5))
                sns.set(style="whitegrid")  # Add gray grid
                sns.scatterplot(
                    x=df.index,
                    y=df[column],
                    alpha=0.5,
                    label="Scatter Plot",
                    palette="Paired",
                    edgecolor=None,
                )
                plt.title(f"Scatter Plot of {column}")
                plt.grid(
                    color="lightgray", linestyle="--", linewidth=0.5
                )  # Add grid with light gray color and dashed lines

                # Save the plot as an image if folder_path is provided
                if folder_path:
                    image_path = os.path.join(folder_path, f"{column}_scatter_plot.png")
                    plt.savefig(image_path, bbox_inches="tight")

                plt.show()
            except:
                pass
