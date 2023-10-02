import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_corr(df, target_col, methods=['pearson', 'spearman', 'kendall'], num_top_features=10):
    """
    Visualizes the major correlations within a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to calculate correlations from.
        target_col (str): The name of the target column for which major correlations will be calculated.
        methods (list): A list of methods to calculate correlations. Default is ['pearson', 'spearman', 'kendall'].
        num_top_features (int): The number of top features with the highest absolute correlation values to display. Default is 10.

    Returns:
        None

    Example:
        # Example usage
        >>> data = {
        ...     'Feature1': [1, 2, 3, 4, 5],
        ...     'Feature2': [5, 4, 3, 2, 1],
        ...     'Feature3': [2, 2, 2, 2, 2],
        ...     'Target': [10, 20, 30, 40, 50]
        ... }
        >>> df = pd.DataFrame(data)
        >>> draw_correlations(df, target_col='Target', methods=['pearson', 'spearman'], num_top_features=2)
        # This will display correlation heatmaps for 'Target' with 'Feature1' and 'Feature2' using Pearson and Spearman methods.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
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
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=False)
        plt.title(f"{method.capitalize()} Correlation with {target_col} (Top {num_top_features})")
        plt.show()

def scatter_plot_dataframe(df, smiles_col="SMILES"):
    """
    Creates scatter plots for each column in the DataFrame against the index.

    Parameters:
        df (DataFrame): The DataFrame containing the data
    """
    for column in df.columns:
        if column in [smiles_col,"id"]:
            pass
        else:
            try:
                plt.figure(figsize=(10, 5))
                plt.scatter(df.index, df[column], label=column)
                plt.title(f"Scatter Plot of {column}")
                plt.xlabel("Index")
                plt.ylabel(column)
                plt.legend()
                plt.show()
            except:
                pass
