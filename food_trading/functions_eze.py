import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def drop_columns_nan(dataframe, threshold: int) -> pd.DataFrame:
  """Drops columns from a DataFrame that contain a certain amount of missing values (NaNs).

  Args:
      dataframe: The pandas DataFrame to process.
      threshold: The minimum number of NaNs a column must have to be dropped.
          A column will be dropped if the number of NaNs is greater than or equal to this threshold.

  Returns:
      The modified DataFrame with columns containing a high number of NaNs dropped.
  """
  columns_to_drop = [col for col in dataframe.columns if dataframe[col].isnull().sum() >= threshold]
  dataframe.drop(columns_to_drop, axis=1, inplace=True)

####################################################################################################

def drop_notinformative(df, threshold=0.95):
    """  drop columns with high percentage of ONE UNIQUE value
    """
    
    columns_to_drop = []
    threshold = 0.9
    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True)
        if len(value_counts) == 1 or value_counts.iloc[0] > threshold:  # Check for single value or exceeding threshold
            columns_to_drop.append(col)

    df.drop(columns_to_drop, axis=1, inplace=True)
  

#################################################################################################

def impute_nan_custom(df, method_numerical='mean', value_categorical='Non-existent', fill_categorical_with_mode=False):
  """
  Imputes missing values in a DataFrame with user-defined options

  Args:
      df (pandas.DataFrame): The DataFrame to impute
      method_numerical (str, optional): Method to impute numerical columns. Defaults to 'mean'. Valid options are 'mean', 'median', 'zero', or 'mode'.
      value_categorical (str, optional): Value to use for imputing missing values in categorical columns. Defaults to 'Non-existent'.
      fill_categorical_with_mode (bool, optional): If True, fills missing values in categorical columns with the mode. Defaults to False.

  Returns:
      pandas.DataFrame: The DataFrame with imputed missing values
      list: A list of columns containing missing values

  Raises:
      ValueError: If an invalid method_numerical is provided.
  """
  columns_with_nan = []
  for col in df.columns:
    if df[col].isnull().sum() > 0 and df[col].dtype.kind in ['f', 'i']:  # Check for float or integer
      if method_numerical == 'mean':
        df[col] = df[col].fillna(value=df[col].mean())
      elif method_numerical == 'median':
        df[col] = df[col].fillna(value=df[col].median())
      elif method_numerical == 'zero':
        df[col] = df[col].fillna(value=0)
      elif method_numerical == 'mode':
        df[col] = df[col].fillna(value=df[col].mode()[0])
      else:
        raise ValueError(f"Invalid method_numerical: {method_numerical}. Valid options are 'mean', 'median', 'zero', or 'mode'.")
      columns_with_nan.append(col)
    elif df[col].isnull().sum() > 0 and fill_categorical_with_mode and pd.api.types.is_categorical_dtype(df[col]):
      df[col] = df[col].fillna(value=df[col].mode()[0])  # Fill with mode for categorical
      columns_with_nan.append(col)
    elif df[col].isnull().sum() > 0:
      df[col] = df[col].fillna(value=value_categorical)
      columns_with_nan.append(col)
  return columns_with_nan

#################################################################################################

def replace_less_frequent(df: pd.DataFrame, list_col: list[str], threshold: float = 0.02, new_value='other') -> pd.DataFrame:
  """Replaces less frequent values in specified columns of a DataFrame with a new value.

  This function identifies values that appear less frequently than a certain threshold
  within specified columns of a DataFrame and replaces them with a new user-defined value.
  It also prints value counts to confirm the changes.

  Args:
      df: The pandas DataFrame to process.
      list_col: A list of column names to be processed.
      threshold: The minimum frequency (between 0 and 1) a value must have to be considered
          frequent. Values with frequency less than the threshold will be replaced.
      new_value: The value to use for replacing less frequent values.

  Returns:
      The modified DataFrame with less frequent values replaced.

  Prints:
      Value counts for each modified column after the replacement.
  """

  vals_to_change = []

  # Iterate over each column in the list
  for col in list_col:
    # Get values with frequency less than the threshold
    filtered_values = df[col].value_counts(normalize=True)
    filtered_values = filtered_values[filtered_values < threshold].index.tolist()
    vals_to_change.extend(filtered_values)  # Extend to avoid nested lists

  # Replace less frequent values with new_value
  for col in list_col:
    df[col] = np.where(df[col].isin(vals_to_change), new_value, df[col])

  # Print value counts for each modified column
  for col in list_col:
    print(f"\nValue Counts for {col} after replacement:")
    print(df[col].value_counts(normalize=True, dropna=False))
    print(f"\n** NEW {col} created correctly**")


#######################################################################################################

def get_categorical(df):
  """
  Identifies categorical columns in a DataFrame

  Args:
      df (pandas.DataFrame): The DataFrame to identify categorical columns in

  Returns:
      list: A list containing the names of categorical columns
  """
  l_cat = []
  for col in df.columns:
    if df[col].dtype.kind == 'O':  # Check for object dtype (categorical)
      l_cat.append(col)
  return l_cat

#######################################################################################################

def get_numeric(df):
  """
  Identifies categorical columns in a DataFrame

  Args:
      df (pandas.DataFrame): The DataFrame to identify categorical columns in

  Returns:
      list: A list containing the names of categorical columns
  """
  l_num = []
  for col in df.columns:
    if df[col].dtype.kind == 'f' or df[col].dtype.kind == 'i':  # Check for object dtype (categorical)
      l_num.append(col)
  return l_num

#######################################################################################################

def get_posible_bool(df):
  """
  Identifies categorical columns in a DataFrame

  Args:
      df (pandas.DataFrame): The DataFrame to identify categorical columns in

  Returns:
      list: A list containing the names of categorical columns
  """
  l_bool = []
  for col in df.columns:
    if len(df[col].unique()) == 2:  # Check for object dtype (categorical)
      l_bool.append(col)
  return l_bool

#######################################################################################################

def transform_dates(df):
    """
    Extract features from datetime columns in a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list of strings): The names of the datetime columns to extract features from.

    Returns:
        pandas.DataFrame: The DataFrame with the extracted features.
    """

    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
          # Extract hour, day of the week, day, and month
          df[col + '_hour'] = df[col].dt.hour
          df[col + '_year'] = df[col].dt.year
          df[col + '_day'] = df[col].dt.day
          df[col + '_month'] = df[col].dt.month

    return df

#######################################################################################################

def awesome_plots(df, target, figsize=(4, 3), palette='Set2'):
    """
    Create violin plots for categorical variables and histograms for numeric variables.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        target (str): The target variable to be plotted against.
        figsize (tuple, optional): Figure size. Default is (10, 6).
        palette (str or list of colors, optional): Color palette for the plots. Default is 'Set2'.

    Returns:
        None
    """
    categorical_cols = get_categorical(df)
    numeric_cols = get_numeric(df)

    for col in categorical_cols:
        g = sns.catplot(data=df, x=col, y=target, kind='violin', palette=palette)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Violin plot of {col} vs {target}')
        plt.show()

    for col in numeric_cols:
        g = sns.displot(data=df, x=col, kde=True, palette=palette)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Histogram of {col}')
        plt.show()