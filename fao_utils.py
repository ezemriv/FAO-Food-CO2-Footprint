import math
import random
from datetime import date
from typing import List, Optional
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

class MLForecast_Evaluator:
    def __init__(self, fcst, valid, future_df, h=1):
        self.fcst = fcst
        self.valid = valid
        self.future_df = future_df
        self.h = h
        self.mean_mape_valid = None  # To use for filename (mean rmse of all models)
        
        # Generate predictions
        self.predictions = fcst.predict(h=self.h, X_df=future_df)
        # Replace negative predictions with 0
        numeric_cols = self.predictions.select_dtypes(include=[np.number])
        numeric_cols[numeric_cols < 0] = 0
        self.predictions[numeric_cols.columns] = numeric_cols

        # Merge predictions with valid set
        self.results = valid.merge(self.predictions, on=['unique_id', 'ds'])
        
        # Get fitted values (train predictions)
        self.train_preds = fcst.forecast_fitted_values()
        
        # Identify model columns
        self.model_columns = list(fcst.models.keys())

    def plot_time_series(self, n_samples: Optional[int] = None, figsize: tuple = None, random_state: Optional[int] = None):
        """
        Plots the time series for a random sample of unique_ids or all if n_samples is not passed.
        
        Parameters
        ----------
        """
        
        # Define a color map for models
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.model_columns)))
        color_map = dict(zip(self.model_columns, colors))

        # Sample random unique_ids
        unique_ids = self.train_preds['unique_id'].unique()
        
        if n_samples is None:
            sampled_ids = unique_ids
        else:
            sampled_ids = np.random.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)
        
        n_samples = len(sampled_ids)
        
        # Calculate grid dimensions
        n_cols = math.ceil(math.sqrt(n_samples))
        n_rows = math.ceil(n_samples / n_cols)
        
        # Calculate adaptive figsize if not provided
        if figsize is None:
            figsize = (7 * n_cols, 4 * n_rows)
        
        # Create subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Time Series Predictions on Train/Valid Data', fontsize=16)

        axs = axs.flatten() if n_samples > 1 else [axs]

        # List to store handles and labels for the legend
        handles, labels = [], []

        for i, unique_id in enumerate(sampled_ids):
            train_data = self.train_preds[self.train_preds['unique_id'] == unique_id]
            valid_data = self.results[self.results['unique_id'] == unique_id]

            # Plot train data
            h_train, = axs[i].plot(train_data['ds'], train_data['y'], label='Actual (Train)', color='black')
            if 'Actual (Train)' not in labels:
                handles.append(h_train)
                labels.append('Actual (Train)')

            for model in self.model_columns:
                # Model predictions on train data (dashed line)
                h_model_train, = axs[i].plot(train_data['ds'], train_data[model], label=f'{model} (Train)', color=color_map[model])
                if f'{model} (Train)' not in labels:
                    handles.append(h_model_train)
                    labels.append(f'{model} (Train)')
            
            # Plot valid data
            h_valid, = axs[i].plot(valid_data['ds'], valid_data['y'], label='Actual (Valid)', color='black', linestyle='--')
            if 'Actual (Valid)' not in labels:
                handles.append(h_valid)
                labels.append('Actual (Valid)')

            for model in self.model_columns:
                # Model predictions on valid data (solid line)
                h_model_valid, = axs[i].plot(valid_data['ds'], valid_data[model], label=f'{model} (Valid)', linestyle='--', color=color_map[model])
                if f'{model} (Valid)' not in labels:
                    handles.append(h_model_valid)
                    labels.append(f'{model} (Valid)')

            axs[i].set_title(f'Region: {unique_id}')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Value')
        
        # Remove any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Add a single legend for all subplots
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=len(self.model_columns) + 2, fontsize='small')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend
        plt.show()

    def calculate_metrics(self) -> pd.DataFrame:
        metrics = {}

        for model in self.model_columns:
            model_metrics = {}
            for result_df, name in zip([self.train_preds, self.results], ['train', 'valid']):
                y_true = result_df['y']
                y_pred = result_df[model]

                model_metrics[f'MAPE_{name}'] = mean_absolute_percentage_error(y_true, y_pred)

            metrics[model] = model_metrics

        metrics_df = pd.DataFrame(metrics).T

        # Find the model with the lowest validation MAPE
        lowest_mape_model = metrics_df['MAPE_valid'].idxmin()
        lowest_mape_value = metrics_df.loc[lowest_mape_model, 'MAPE_valid']
        
        self.mean_mape_valid = metrics_df['MAPE_valid'].mean()  # Calculate and store mean MAPE for submission

        print(f"MEAN MAPE_VALID = {self.mean_mape_valid*100:.2f}%\n")
        print(f"Model with lowest MAPE validation is {lowest_mape_model} with MAPE = {lowest_mape_value*100:.2f}%\n")

        print(metrics_df.sort_values(by='MAPE_valid'))

        return metrics_df.sort_values(by='MAPE_valid')
    
    def plot_feature_importances(self):
        # Initialize an empty DataFrame to store the feature importances
        df = pd.DataFrame()

        # Loop through each model to get its feature importances
        for model in self.model_columns:
            feature_importances = self.fcst.models_[model].feature_importances_
            feature_names = self.fcst.ts.features_order_

            # Create a temporary DataFrame for the current model
            temp_df = pd.DataFrame(feature_importances, columns=[model], index=feature_names)

            # Merge the temporary DataFrame with the main DataFrame
            if df.empty:
                df = temp_df
            else:
                df = df.join(temp_df, how='outer')

        # Fill NaNs with 0 (if any feature is missing in some models)
        df = df.fillna(0)
        # Scale the feature importances between 0 and 1 for each model
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

        # Sort features by their average importance
        average_importance = df_scaled.mean(axis=1)
        sorted_features = average_importance.sort_values(ascending=True).index[-20:]

        # Reorder DataFrame according to the sorted feature list
        df_scaled = df_scaled.loc[sorted_features]

        # Plotting the horizontal multi-bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Define the height of the bars and the positions for each group
        bar_height = 0.15
        index = np.arange(len(sorted_features))

        # Loop through each model and plot its feature importances
        for i, model in enumerate(self.model_columns):
            ax.barh(index + i * bar_height, df_scaled[model], bar_height, label=model)

        # Add labels, title, and legend
        ax.set_ylabel('Feature')
        ax.set_xlabel('Scaled Importance')
        ax.set_title('Top 20 Scaled Feature Importances by Model')
        ax.set_yticks(index + bar_height * (len(self.model_columns) - 1) / 2)
        ax.set_yticklabels(sorted_features)
        ax.legend()

        plt.tight_layout()
        plt.show()


#----------------------------------------------------------------------------#

def query_country(df, unique_id=None, target_col=None):
    """
    This function queries a DataFrame based on a unique identifier and optionally plots a time series of a specified target column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to query.
    unique_id (str, optional): The unique identifier to filter the DataFrame. If not provided, a random unique identifier will be chosen.
    target_col (str, optional): The target column to plot as a time series. If not provided, no plot will be generated.

    Returns:
    pandas.DataFrame: The queried DataFrame. If no data is found for the specified unique identifier, an empty DataFrame is returned.
    """
    if unique_id:
        query_df = df[df['unique_id'] == unique_id]
    else:
        unique_id = random.choice(df['unique_id'].unique())
        query_df = df[df['unique_id'] == unique_id]

    if query_df.empty:
        print(f"No data found for unique_id: {unique_id}")
        return pd.DataFrame()  # Return an empty DataFrame explicitly
    elif target_col:
        if target_col in query_df.columns:
            sns.lineplot(data=query_df, x='ds', y=target_col)
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Column '{target_col}' does not exist in the DataFrame.")
            return query_df

    return query_df

#----------------------------------------------------------------------------#

def load_fao_table(table_number, path):
    
    # Loop through all files in the directory
    for filename in os.listdir(path):
        # Extract the number from the start of the filename
        file_number_str = filename.split('-')[0]
        try:
            file_number = int(file_number_str)
            # Check if the number matches the table number
            if file_number == table_number:
                full_path = os.path.join(path, filename)

                if filename.endswith(".zip"):
                    # Open the zip file
                    with zipfile.ZipFile(full_path, 'r') as zip_ref:
                        # Iterate through the file names in the zip archive
                        for file_name in zip_ref.namelist():
                            # Check if the file name contains the pattern "All_Data_" and ends with .csv
                            if "All_Data" in file_name and file_name.endswith(".csv"):
                                # Read the CSV file into a DataFrame
                                with zip_ref.open(file_name) as file:
                                    return pd.read_csv(file, encoding="ISO-8859-1")

                elif filename.endswith(".csv"):
                    # Read the CSV file into a DataFrame
                    return pd.read_csv(full_path, encoding="ISO-8859-1")

        except ValueError:
            # Handle the case where the filename doesn't start with a valid number
            continue
    
    # Return None if no matching file is found
    return None