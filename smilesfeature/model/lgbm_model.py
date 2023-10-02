import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.font_manager as fm


def train_lgbm(df, target_col, hyperparameter_tuning="off", param_grid={}):
    """
    Train a LightGBM classifier or regressor, perform hyperparameter tuning if specified, and visualize results.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing both features and target column.
        target_col (str): The name of the target column.
        hyperparameter_tuning (str, optional): "on" to perform hyperparameter tuning, "off" otherwise. Default is "off".
        param_grid (dict, optional): Hyperparameter grid for GridSearchCV. Default is an empty dictionary.

    Returns:
        model: The trained LightGBM model.

    Example:
        target_col = 'pIC50'  # or any other target column
        hyperparameter_tuning = "on"  # "on" or "off" to enable or disable hyperparameter tuning
        param_grid = {}  # Define your hyperparameter grid here if needed
        model = train_lgbm(df_test, target_col, hyperparameter_tuning, param_grid)
    """
    
    
    df.dropna(how="any", inplace=True)
    # Extract numeric columns
    numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    
    # Remove the target column from the features
    numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Check if the target column is categorical or continuous
    is_cat = pd.api.types.is_categorical_dtype(df[target_col]) or len(df[target_col].unique()) < 20

    # Initialize model based on y's type
    if is_cat:
        print("The target column consists of categorical values, so we should use a classifier.")
        if hyperparameter_tuning == "on":
            print("Performing hyperparameter tuning for the classifier.")
            if not param_grid:
                param_grid = {
                    # Define hyperparameter grid for the classifier
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    # Add more hyperparameters as needed
                }
            model = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=3)
        else:
            model = lgb.LGBMClassifier()
    else:
        print("The target column consists of continuous values, so we should use a regressor.")
        if hyperparameter_tuning == "on":
            print("Performing hyperparameter tuning for the regressor.")
            if not param_grid:
                param_grid = {
                    # Define hyperparameter grid for the regressor
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    # Add more hyperparameters as needed
                }
            model = GridSearchCV(lgb.LGBMRegressor(), param_grid, cv=3)
        else:
            model = lgb.LGBMRegressor()

    # Fit the model, handling possible exceptions
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Using LGMB regressor instead.")
        model = lgb.LGBMRegressor()
        model.fit(X_train, y_train)

    # Predict using the test set
    y_pred = model.predict(X_test)
    
    # Scatter plot with Seaborn
    plt.figure(figsize=(8, 8))
    
    if is_cat:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {accuracy:.2f}")
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, label='Scatter Plot', palette='Paired')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='y=x', color='gray')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title(f"Scatter plot of Actual vs Predicted {target_col} - Classification")
        
    else:
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE on test set: {mse:.2f}")
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, label='Scatter Plot', palette='Paired')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='y=x', color='gray')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title(f"Scatter plot of Actual vs Predicted {target_col} - Regression")

        # Histogram with Seaborn
        plt.figure(figsize=(8, 6))
        sns.histplot(y_test - y_pred, color='purple', alpha=0.5, kde=True, palette='Paired')
        plt.xlabel(f"Actual - Predicted ({target_col})")
        plt.ylabel('Count')
        plt.title(f"Histogram of Residuals (Actual - Predicted)")

        # Q-Q Plot
        plt.figure(figsize=(16, 5))
        for i in range(1, 4):
            plt.subplot(1, 3, i)
            stats.probplot(y_test - y_pred, plot=plt)
            plt.title(f"Q-Q Plot {i} of Residuals (Actual - Predicted)")

    plt.show()

    return model
