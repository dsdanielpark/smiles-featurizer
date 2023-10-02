import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_error(true_col: pd.Series, predicted_col: pd.Series, metric: str) -> float:
    """
    Calculate the error between true and predicted values using various metrics.

    Parameters:
        true_col (pd.Series): True values.
        predicted_col (pd.Series): Predicted values.
        metric (str): The metric to use for error calculation. Options are 'RMSE' (Root Mean Squared Error),
            'MAE' (Mean Absolute Error), and 'R2' (R-squared).

    Returns:
        float: The calculated error value.
    """
    if metric == 'RMSE':
        return np.sqrt(mean_squared_error(true_col, predicted_col))
    elif metric == 'MAE':
        return mean_absolute_error(true_col, predicted_col)
    elif metric == 'R2':
        return r2_score(true_col, predicted_col)
    else:
        return None

def save_smiles_analysis_plot(df: pd.DataFrame, folder: str, true_col: str, predicted_col: str, metric: str, idx_start: int = 1):
    """
    Generate and save analysis plots for SMILES data.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the SMILES data and additional information.
        folder (str): The folder where the analysis plots will be saved.
        true_col (str): The column name for the true values.
        predicted_col (str): The column name for the predicted values.
        metric (str): The metric to display in the plots. Options are 'RMSE' (Root Mean Squared Error),
            'MAE' (Mean Absolute Error), and 'R2' (R-squared).
        idx_start (int, optional): The starting index for numbering saved plots (default is 1).

    Example usage:
        selected_metric = 'RMSE'  # Choose the error metric you want to display
        true_column = 'pIC50'  # Replace with your true column name
        predicted_column = 'predicted_pIC50'  # Replace with your predicted column name
        save_smiles_analysis_plot(df[:1], 'output_folder', true_column, predicted_column, selected_metric)
    """
    for idx, (index, row) in enumerate(df.iterrows(), start=idx_start):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

        # SMILES and molecular formula as title
        smiles = row.get('SMILES', 'Unknown')
        molecule = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        axs[1].set_title(f"Molecule: {formula}\nSMILES: {smiles}")

        # Subplot for additional_info
        additional_info = row['dm_descriptor_dict']
        info_text = '\n'.join([f"{key}: {value:.2f}" for key, value in additional_info.items()])
        axs[0].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', horizontalalignment='left')
        axs[0].axis('off')

        # Subplot for the image_array
        axs[1].imshow(row['image_array'])
        axs[1].axis('off')

        # Calculate the selected error metric
        error = calculate_error([row[true_col]], [row[predicted_col]], metric)

        # Bar plot for the selected metric on the right
        metric_names = [f'True {metric}', f'Predicted {metric}']
        metric_errors = [row[true_col], row[predicted_col]]
        axs[2].bar(metric_names, metric_errors, color=['blue', 'red'])
        axs[2].text(0, row[true_col] * 1.05, f"True {metric}: {row[true_col]:.2f}", color='blue', fontsize=14)
        axs[2].text(1, row[predicted_col] * 1.05, f"Predicted {metric}: {row[predicted_col]:.2f}", color='red', fontsize=14)
        axs[2].text(0.5, max(row[true_col], row[predicted_col]) * 0.8, f"{metric} Error: {error:.2f}", color='black', fontsize=14)
        axs[2].set_ylabel('Error')
        axs[2].set_title(f'{predicted_col} {metric} Error')

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{folder}/{idx}.jpg", dpi=300, bbox_inches='tight', pad_inches=0.1 )
        plt.clf()
