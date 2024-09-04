#from google.colab import drive
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import scipy.stats as stats
import numpy as np
from tabulate import tabulate
from IPython.display import display, HTML
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import (
    cross_val_score, LeaveOneOut, StratifiedKFold, TimeSeriesSplit, RepeatedKFold,
    LeavePOut, KFold, GroupKFold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Global Variables
numeric_cols = []
categorical_cols = []
object_cols = []
datetime_cols = []
boolean_cols = []

def file_to_df(file_path):
    """
    Reads a file into a DataFrame based on its extension.
    Supported file types are: csv, json, txt, xls, xlsx.
    """
    # Check the file extension and read the file accordingly
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == '.txt':
        # Assuming the TXT file is tabular data separated by commas
        return pd.read_csv(file_path, delimiter='\t')
    else:
        raise ValueError("Unsupported file type")

"""### **Basic EDA**"""

def first_look(df):
    from IPython.display import display

    # İlk 5 satırı gösterir
    print("### İlk 5 Satır:")
    display(df.head())

    # Veri çerçevesi hakkında bilgi verir
    print("\n### Veri Çerçevesi Bilgisi:")
    display(df.info())

    # Eksik değerleri gösterir
    print("\n### Eksik Değerlerin Toplamı:")
    display(df.isna().sum())

    # İstatistiksel özet verir
    print("\n### İstatistiksel Özet:")
    display(df.describe())

    # Yinelenen satırların sayısını gösterir
    print("\n### Yinelenen Satırların Toplamı:")
    display(df.duplicated().sum())


def eda(df):
    print("Basic Information:\n")
    print(df.info())
    print("\nSummary Statistics:\n")
    print(df.describe())
    print("\nFirst Few Rows:\n")
    display(HTML(df.head().to_html()))
    print("\nVisualizations:\n")

    # Numeric columns for histogram and Q-Q plots
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        # Create a 2x1 subplot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Histogram
        sns.histplot(df[col], kde=True, ax=axs[0])
        axs[0].set_title(f'Histogram of {col}')

        # Q-Q Plot
        stats.probplot(df[col].dropna(), dist="norm", plot=axs[1])
        axs[1].set_title(f'Q-Q Plot of {col}')

        # Show the plots
        plt.show()

        # Analysis based on skewness and kurtosis
        data = df[col].dropna()
        if data.empty:
            print(f"Analysis for {col}: Undefined\n")
            continue

        skewness = data.skew()
        kurtosis = data.kurtosis()

        analysis_text = f"Analysis for {col}:\n"
        if skewness > 1 or skewness < -1:
            analysis_text += f"  Highly skewed distribution. Skewness: {skewness}\n"
        elif skewness >= 0.5 or skewness <= -0.5:
            analysis_text += f"  Moderately skewed distribution. Skewness: {skewness}\n"
        else:
            analysis_text += f"  Symmetric distribution. Skewness: {skewness}\n"

        if kurtosis > 3:
            analysis_text += f"  Leptokurtic distribution (sharp peak, heavy tails). Kurtosis: {kurtosis}\n"
        elif kurtosis < 3:
            analysis_text += f"  Platykurtic distribution (broad peak, light tails). Kurtosis: {kurtosis}\n"
        else:
            analysis_text += f"  Mesokurtic distribution (normal distribution). Kurtosis: {kurtosis}\n"

        print(analysis_text)

    # Visualizations for categorical columns
    for column in df.select_dtypes(include='object').columns:
        sns.countplot(y=column, data=df)
        plt.show()

    # Correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        print("\nCorrelation Matrix:\n")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

"""### **Categorise Columns**"""

def categorize_columns(df):
    """
    Function to identify and categorize columns of a DataFrame by their data types.
    It categorizes numeric, categorical, object, datetime, timedelta, and boolean columns
    and assigns them to global variables.

    :param df: DataFrame to analyze.
    """
    global numeric_cols, categorical_cols, object_cols, datetime_cols, boolean_cols

    # Identifying numeric, categorical, object, datetime, timedelta, and boolean columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

    # Printing the lists as comma-separated strings
    print("numeric_cols:", ", ".join(['"' + col + '"' for col in numeric_cols]))
    print("categorical_cols:", ", ".join(['"' + col + '"' for col in categorical_cols]))
    print("object_cols:", ", ".join(['"' + col + '"' for col in object_cols]))
    print("datetime_cols:", ", ".join(['"' + col + '"' for col in datetime_cols]))
    print("boolean_cols:", ", ".join(['"' + col + '"' for col in boolean_cols]))

    # Returning the lists for further use
    #return numeric_cols, categorical_cols, object_cols, datetime_cols, boolean_cols


def list_unique(df):

    """
    Lists unique values in every column of a DataFrame, excluding ID columns, sorted alphabetically (A to Z),
    and displays them in a formatted manner where each column's unique values are listed separately.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        None: Prints formatted unique values for each column.
    """
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is an ID column (all values are unique)
        if df[col].nunique() == len(df[col]):
            print(f"'{col}' column identified as ID column and excluded from listing.")
            continue  # Skip ID columns
        
        # Get unique values from the column and sort them
        unique_values = sorted(df[col].unique(), key=lambda x: str(x).lower())
        
        # Format the unique values as a comma-separated string
        formatted_values = ', '.join(str(value) for value in unique_values)
        
        # Print the formatted output for the column
        print(f"{col}: {formatted_values}")


def bar_plots(df):
    """
    Creates a bar plot for each column in a DataFrame, excluding ID columns (columns with all unique values).
    The plots are wide and aesthetically pleasing.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        None: Displays the bar plots for each non-ID column.
    """
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is an ID column (all values are unique)
        if df[col].nunique() == len(df[col]):
            print(f"'{col}' column identified as ID column and excluded from plotting.")
            continue  # Skip ID columns
        
        # Count the frequency of each unique value in the column
        value_counts = df[col].value_counts().sort_index()
        
        # Create a bar plot for the column
        plt.figure(figsize=(12, 6))  # Make the plot wide
        value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        
        # Add title and labels
        plt.title(f'Bar Plot for {col}', fontsize=16)
        plt.xlabel(f'{col} Values', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
        
        # Improve layout
        plt.tight_layout()
        
        # Display the plot
        plt.show()


def detect_outliers(df):
    outlier_indices = []

    # Iterate over each column
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find indices of outliers
        outlier_list_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

        # Append the found outlier indices for the column
        outlier_indices.extend(outlier_list_col)

    # Select observations containing more than 2 outliers
    outlier_indices = list(set(outlier_indices))
    return df.loc[outlier_indices]

def detect_outliers_inlist(df, columns):
    outlier_indices = []

    # Iterate over each specified column
    for col in columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Find indices of outliers
            outlier_list_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

            # Append the found outlier indices for the column
            outlier_indices.extend(outlier_list_col)
        else:
            print(f"Column '{col}' is not in DataFrame or is not a numeric column.")

    # Select observations containing outliers
    outlier_indices = list(set(outlier_indices))
    return df.loc[outlier_indices]

"""### **List Outliers**"""

def list_outliers(df, column):
    if df[column].dtype in ['float64', 'int64']:
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Determine outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        lower_outliers = df[df[column] < lower_bound]
        upper_outliers = df[df[column] > upper_bound]

        # Create a DataFrame to display as HTML
        outliers_df = pd.DataFrame({
            'Lower Outliers': lower_outliers[column].reset_index(drop=True),
            'Upper Outliers': upper_outliers[column].reset_index(drop=True)
        })

        # Display the DataFrame as HTML
        display(HTML(outliers_df.to_html()))
    else:
        display(HTML(f"<p>Column '{column}' is not numerical. Please provide a numerical column.</p>"))

"""### **Remove Outliers**"""

def remove_outliers(df, columns):
    clean_df = df.copy()
    outlier_counts = {}

    # Iterate over each specified column
    for col in columns:
        if col in clean_df.columns and clean_df[col].dtype in ['float64', 'int64']:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Find indices of outliers
            outliers = clean_df[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)

            # Remove outliers
            clean_df = clean_df.drop(outliers.index)
        else:
            print(f"Column '{col}' is not in DataFrame or is not a numeric column.")

    # Print the number of outliers removed from each column
    for col, count in outlier_counts.items():
        print(f"{count} outliers removed from '{col}'.")

    return clean_df

def histogram_qq(df):
    # Select only the numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Plotting histograms and Q-Q plots for each numeric column
    for col in numeric_cols:
        # Create a 2x1 subplot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Histogram
        sns.histplot(df[col], kde=True, ax=axs[0])
        axs[0].set_title(f'Histogram of {col}')

        # Q-Q Plot
        stats.probplot(df[col].dropna(), dist="norm", plot=axs[1])
        axs[1].set_title(f'Q-Q Plot of {col}')

        # Show the plots
        plt.show()

"""### **Find and Replace**"""

import pandas as pd

def find_and_replace(df, to_find, to_replace, columns=None, exact=True):
    """
    
    Find and replace a value in a DataFrame, optionally in specific columns, and print the count of replacements per column.

    :param df: DataFrame in which the operation is performed.
    :param to_find: Value to find.
    :param to_replace: Value to replace with.
    :param columns: Optional. A column name or list of column names to limit the operation.
    :param exact: Optional. If True, performs an exact match replacement. If False, replaces any matching part of a string.
    :return: Modified DataFrame.
    """
    if columns is None:
        columns = df.columns.tolist()

    if not isinstance(columns, list):
        columns = [columns]

    for column in columns:
        # Apply operations only to columns of appropriate data type
        if df[column].dtype.kind in ['O', 'U', 'S']:  # Object, Unicode (string), String
            if exact:
                initial_count = df[column].eq(to_find).sum()
                df[column] = df[column].replace(to_find, to_replace, regex=False)
                final_count = df[column].eq(to_find).sum()
            else:
                # For non-exact match, count the rows containing 'to_find' before replacement
                initial_count = df[column].str.contains(to_find, na=False).sum()
                df[column] = df[column].str.replace(to_find, to_replace, regex=True)
                # Count the rows containing 'to_find' after replacement
                final_count = df[column].str.contains(to_find, na=False).sum()

            replaced_count = initial_count - final_count
            if replaced_count > 0:
                print(f"Replaced {replaced_count} row(s) in column '{column}'.")

"""### **List Null**"""

def listnull(df):
    """
    Plots two bar charts:
    1. Showing the number of null values in each column of the DataFrame, sorted in descending order.
    2. Showing the percentage of null values in each column of the DataFrame, sorted in descending order.

    :param df: pandas DataFrame
    """
    # Counting null values and calculating percentages
    null_counts = df.isna().sum()
    null_percentage = round((null_counts / len(df)) * 100, 2)

    # Sorting in descending order
    null_counts_sorted = null_counts.sort_values(ascending=False)
    null_percentage_sorted = null_percentage.sort_values(ascending=False)

    # Plot for absolute null values count
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    null_counts_sorted.plot(kind='bar', ax=ax1)
    plt.ylabel('Null Values Count')
    plt.title('Null Values Count in Each Column (Sorted)')
    plt.xticks(rotation=45)

    # Adding count above each bar in the first plot
    for p in ax1.patches:
        ax1.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    # Plot for percentage of null values
    ax2 = plt.subplot(1, 2, 2)
    null_percentage_sorted.plot(kind='bar', ax=ax2, color='orange')
    plt.ylabel('Percentage of Null Values')
    plt.title('Percentage of Null Values in Each Column (Sorted)')
    plt.xticks(rotation=45)

    # Adding percentage above each bar in the second plot
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}%", (p.get_x() * 1.005, p.get_height() * 1.005))

    # Adjust layout
    plt.tight_layout()
    plt.show()

"""### **Handle Null Values**"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

def handle_null_values(df, columns, method='drop', fill_value=None, statistic=None, target_column=None, n_neighbors=5):

    """
    Handle and/or impute null values in specified columns of the DataFrame using various methods including regression and KNN.

    :param df: DataFrame to handle/impute null values in.
    :param columns: List of columns to handle/impute null values in.
    :param method: Method to handle/impute null values ('fill', 'drop', 'statistic', 'regression', 'knn').
    :param fill_value: Value to fill nulls with if method is 'fill'.
    :param statistic: Statistic to fill nulls with ('mean', 'median', 'mode') if method is 'statistic'.
    :param target_column: Target column for regression if method is 'regression'.
    :param n_neighbors: Number of neighbors for KNN imputation.
    :return: DataFrame with handled/imputed null values in specified columns.
    """

    for col in columns:
       initial_null_count = df[col].isnull().sum()
       if initial_null_count == 0:
           print(f"No null values to handle in '{col}'.")
           continue

       filled_values = set()
       if method == 'fill':
           df[col] = df[col].fillna(fill_value)
           filled_values.add(fill_value)
       elif method == 'drop':
           df = df.dropna(subset=[col])
       elif method == 'statistic':
           value = None
           if statistic == 'mean':
               value = df[col].mean()
           elif statistic == 'median':
               value = df[col].median()
           elif statistic == 'mode':
               value = df[col].mode()[0]
           df[col] = df[col].fillna(value)
           filled_values.add(value)
       elif method == 'regression':
           df, regression_values = impute_with_regression(df, col, target_column)
           filled_values.update(regression_values)
       elif method == 'knn':
           df, knn_values = impute_with_knn(df, col, n_neighbors)
           filled_values.update(knn_values)
       else:
           raise ValueError("Invalid method. Choose 'fill', 'drop', 'statistic', 'regression', or 'knn'.")

       final_null_count = df[col].isnull().sum()
       nulls_handled = initial_null_count - final_null_count

       filled_values_str = ', '.join(map(str, filled_values)) if filled_values else 'No unique values'
       print(f"{nulls_handled} null values handled in '{col}' (filled unique values: {filled_values_str}).")

    print("\nDataFrame Information after Handling Null Values:")
    df.info()
    return df

def impute_with_regression(df, column, target_column):
    not_null_df = df[df[column].notnull()]
    train_X = not_null_df[[target_column]]
    train_y = not_null_df[column]

    model = LinearRegression()
    model.fit(train_X, train_y)

    null_df = df[df[column].isnull()]
    predicted_values = model.predict(null_df[[target_column]])

    regression_values = set(np.round(predicted_values, decimals=3))  # Rounded for uniqueness
    df.loc[df[column].isnull(), column] = predicted_values

    return df, regression_values

def impute_with_knn(df, column, n_neighbors):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df_numeric)
    df_numeric_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)

    original_values = df_numeric[column].dropna().unique()
    imputed_values = df_numeric_imputed[column].unique()
    knn_values = set(imputed_values) - set(original_values)  # Only new, imputed values

    df[numeric_cols] = df_numeric_imputed[numeric_cols]

    return df, knn_values

# Example usage:
# df = pd.read_csv('your_file.csv')
# columns_to_handle = ['Age', 'OtherColumns']
# df_handled = handle_and_impute_null_values(df, columns_to_handle, method='regression', target_column='Fare')

"""### **Multicollinearity**"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def detect_multicollinearity(df, threshold=5.0):
    """
    Detects multicollinearity in the DataFrame by calculating VIF (Variance Inflation Factor).

    :param df: DataFrame with numeric features.
    :param threshold: VIF threshold to identify features with multicollinearity.
    :return: None
    """
    # Add a constant for VIF calculation
    df_vif = add_constant(df.select_dtypes(include=['float64', 'int64']))

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]

    # Identify features with high VIF
    multicollinear_features = vif_data[vif_data['VIF'] > threshold]

    print("Multicollinear Features and their VIF values:")
    print(multicollinear_features)


def encode(df, columns, method='label'):
    """
    Encode specified columns of a DataFrame using One-Hot Encoding or Label Encoding.

    :param df: DataFrame to encode.
    :param columns: Single column or list of columns to encode.
    :param method: Encoding method ('onehot' for One-Hot Encoding, 'label' for Label Encoding).
    :return: DataFrame with encoded columns.
    """
    df_encoded = df.copy()

    # Convert a single column name to a list
    if isinstance(columns, str):
        columns = [columns]

    if method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        for col in columns:
            print(f"One-Hot Encoding column: {col}")
            encoded_data = encoder.fit_transform(df_encoded[[col]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]), index=df_encoded.index)
            df_encoded = df_encoded.join(encoded_df)
            df_encoded.drop(col, axis=1, inplace=True)
    elif method == 'label':
        encoder = LabelEncoder()
        for col in columns:
            print(f"Label Encoding column: {col}")
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
    else:
        raise ValueError("Invalid method. Choose 'onehot' or 'label'.")

    return df_encoded


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    :param df: DataFrame from which columns are to be dropped.
    :param columns_to_drop: Single column name or list of column names to drop.
    :return: DataFrame with specified columns dropped.
    """
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]

    # Drop columns that are in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    return df.drop(columns=columns_to_drop, axis=1, inplace=True)


"""### **Drop Duplicates**"""

def drop_duplicates(df):
    # Number of rows before removing duplicates
    initial_count = df.shape[0]

    # Removing duplicates
    df = df.drop_duplicates()

    # Number of rows after removing duplicates
    final_count = df.shape[0]

    # Number of duplicates removed
    duplicates_removed = initial_count - final_count

    print(f"Number of duplicates removed: {duplicates_removed}")

"""### **HYpothesis Testing**"""



def hypothesis_test(df, column, group_column, group1, group2, alpha=0.05):
    """
    Perform a T-test to compare the means of two groups within a DataFrame.

    :param df: Pandas DataFrame containing the data.
    :param column: The column on which to perform the test.
    :param group_column: The column used to define groups.
    :param group1: The first group for comparison.
    :param group2: The second group for comparison.
    :param alpha: Significance level, default is 0.05.
    :return: None, prints the test results.
    """

    # Filter data for the two groups
    data1 = df[df[group_column] == group1][column]
    data2 = df[df[group_column] == group2][column]

    # Perform T-test
    statistic, p_value = stats.ttest_ind(data1, data2, nan_policy='omit')

    # Output
    print(f"T-statistic: {statistic}, P-value: {p_value}")

    # Interpretation
    if p_value < alpha:
        print(f"Reject the null hypothesis - significant difference between {group1} and {group2}")
    else:
        print(f"Fail to reject the null hypothesis - no significant difference between {group1} and {group2}")

# Example usage
# df = pd.DataFrame(...)  # Your dataframe
# hypothesis_test(df, 'your_column', 'your_group_column', 'gro

"""### **Train Test Split**"""

import pandas as pd
from sklearn.model_selection import train_test_split

def reduce_dataset(df, yuzde):
    """
    Reduces the size of the DataFrame by randomly selecting a percentage of rows.
    
    :param df: DataFrame to be reduced.
    :param yuzde: Percentage of the dataset to keep (between 0 and 100).
    :return: Reduced DataFrame containing the specified percentage of randomly selected rows.
    """
    if yuzde < 0 or yuzde > 100:
        raise ValueError("Percentage (yuzde) must be between 0 and 100.")

    # Calculate the fraction of the dataset to keep
    fraction = yuzde / 100
    
    # Sample the DataFrame to get the specified percentage of rows
    df_reduced = df.sample(frac=fraction, random_state=42)  # random_state is used for reproducibility

    return df_reduced



def split_data(df, test_size=0.2, validation_size=0.5, random_state=42):
    """
    Split the DataFrame into training, validation, and testing sets.

    :param df: The DataFrame to split.
    :param test_size: Proportion of the data to be used as the initial test set before validation split.
    :param validation_size: Proportion of the initial test set to be used as validation (relative to the test set).
    :param random_state: Random state for reproducibility.
    :return: Three DataFrames: training, validation, and testing sets.
    """

    # Print the parameters that can be adjusted
    print("Parameters you can adjust:")
    print(f"- test_size (default=0.2): {test_size}")
    print(f"- validation_size (default=0.5): {validation_size}")
    print(f"- random_state (default=42): {random_state}\n")

    # Split the dataframe into training and initial test sets
    df_train, df_temp = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the temporary dataframe into validation and final test sets
    df_validation, df_test = train_test_split(df_temp, test_size=validation_size, random_state=random_state)

    return df_train, df_validation, df_test

# Örnek kullanım:
# df = pd.read_csv('path_to_your_data.csv')

# Veri setlerini bölmek için fonksiyonu çağırın
# df_train, df_validation, df_test = split_data(df, test_size=0.4, validation_size=0.5, random_state=42)

"""## **TOOLS**

### **Display Metrics**
"""

def display_metrics(results):
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
    print(results_df)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=results_df.index, y='Value', data=results_df)
    plt.xticks(rotation=45)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Value')
    plt.show()

def display_roc_curve(y_test, y_proba):
    # Plotting ROC Curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def display_pr_curve(y_test, y_proba):
    # Plotting Precision-Recall Curve
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

"""### **Calculate Metrics**"""

def calculate_metrics(y_true, y_pred, y_proba):

    results = {
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
        'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
        'F1-Score': metrics.f1_score(y_true, y_pred, average='macro'),
        'ROC-AUC': metrics.roc_auc_score(y_true, y_proba),
        'Log Loss': metrics.log_loss(y_true, y_proba),
        'Cohen\'s Kappa': metrics.cohen_kappa_score(y_true, y_pred),
        # Add any other metrics you need
    }

    # For Specificity, calculate True Negative Rate
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    results['Specificity'] = specificity

    # Display results in a table using DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
    return results

"""### **Explain**"""

import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test, plot_type='bar', num_features=10, plot_individual=False, individual_index=0):
    """
    Function to explain a CatBoost model using SHAP.

    :param model: The trained CatBoost model.
    :param X_train: Training data (features) used for the model.
    :param X_test: Test data (features) to explain the model predictions.
    :param plot_type: Type of plot for summary ('bar', 'beeswarm'). Default is 'bar'.
    :param num_features: Number of top features to show in the summary plot. Default is 10.
    :param plot_individual: If True, plot SHAP values for an individual prediction. Default is False.
    :param individual_index: Index of the individual instance in X_test to explain. Default is 0.
    """
    # Create the explainer and calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Plot summary plot
    if plot_type == 'bar':
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=num_features)
    elif plot_type == 'beeswarm':
        shap.summary_plot(shap_values, X_test, max_display=num_features)

    # Optionally, plot SHAP values for an individual prediction
    if plot_individual:
        shap.plots.waterfall(shap_values[individual_index], max_display=num_features)

# Example usage:
# explain_model_with_shap(trained_model, X_train, X_test, plot_type='bar', num_features=10, plot_individual=True, individual_index=0)

"""### **Optuna**"""

def objective(trial, X_train, y_train, model_type, cv_method, cv_folds, groups, n_repeats, p, scoring, param_grid):
    params = {}
    for param, values in param_grid.items():
        if isinstance(values, list):  # Categorical parameters
            params[param] = trial.suggest_categorical(param, values)
        elif isinstance(values, tuple) and len(values) == 2:  # Numeric ranges
            if isinstance(values[0], int):
                params[param] = trial.suggest_int(param, values[0], values[1])
            else:
                params[param] = trial.suggest_float(param, values[0], values[1])

    # Choose model based on the model type
    if model_type == 'catboost':
        model = CatBoostClassifier(**params, verbose=False)
    elif model_type == 'xgboost':
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'lightgbm':
        model = LGBMClassifier(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(**params)
    elif model_type == 'regression':  # For regression tasks
        model = RandomForestRegressor(**params)
    # Add other models here as needed

    if cv_method is not None:
        cv = get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        return cv_scores.mean()
    else:
        model.fit(X_train, y_train)
        if model_type in ['random_forest', 'regression']:
            # For regression models, return negative MSE as the objective to minimize
            return -mean_squared_error(y_train, model.predict(X_train))
        else:
            # For classification models, return accuracy as the objective to maximize
            return accuracy_score(y_train, model.predict(X_train))

"""### **Cross-Validation**"""

def get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p):
    if cv_method == 'loocv':
        return LeaveOneOut()
    elif cv_method == 'stratified_kfold':
        return StratifiedKFold(n_splits=cv_folds)
    elif cv_method == 'timeseries':
        return TimeSeriesSplit(n_splits=cv_folds)
    elif cv_method == 'repeated':
        return RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats)
    elif cv_method == 'leave_p_out':
        return LeavePOut(p=p)
    elif cv_method == 'group_kfold':
        if groups is None:
            raise ValueError("Groups must be provided for Group K-Fold CV")
        return GroupKFold(n_splits=cv_folds)
    else:
        return KFold(n_splits=cv_folds)

"""## **MODELS**

### **Logistic Regression**
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_regression(train_df, test_df, target_column):
    """
    Train a logistic regression model and evaluate it on test data.

    :param train_df: Training DataFrame.
    :param test_df: Testing DataFrame.
    :param target_column: The name of the target column.
    :return: None, prints and plots model evaluation metrics.
    """
    # Hedef sütunun veri çerçevesinde mevcut olup olmadığını kontrol edin
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise KeyError(f"Hedef sütun '{target_column}' veri çerçevelerinde bulunamadı. Lütfen sütun adını kontrol edin.")

    # Splitting the train and test data
    X_train = train_df.drop(columns=[target_column], errors='ignore')
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column], errors='ignore')
    y_test = test_df[target_column]

    # Create and fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Model Accuracy: {accuracy}\n")
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")

    # Plotting Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
# Make sure your DataFrame has the target column and the features are all numeric
# logistic_regression(train_df, test_df, 'target_column_name')

"""### **Decision Tree**"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

"""
max_depth: It controls the maximum depth of the tree, limiting the number of nodes and preventing overfitting. Example: max_depth': 3.
min_samples_split: It specifies the minimum number of samples required to split an internal node. Higher values prevent the tree from splitting too aggressively and can help reduce overfitting. Example: min_samples_split': 5.
min_samples_leaf: It sets the minimum number of samples required to be in a leaf node. This can help control the granularity of the tree and reduce overfitting. Example: min_samples_leaf': 2.
max_features: It determines the maximum number of features to consider when making a split. It can be an integer (number of features) or a fraction (percentage of features). Example: max_features': 'sqrt' (square root of the total features).
criterion: It specifies the criterion used to measure the quality of a split. Common values include 'gini' for Gini impurity and 'entropy' for information gain. Example: criterion': 'gini'.
max_leaf_nodes: It limits the maximum number of leaf nodes in the tree. If not set, it allows the tree to grow without any restriction. Example: max_leaf_nodes': 10.
min_impurity_decrease: It sets a threshold for splitting nodes based on impurity decrease. It prevents splits that do not lead to a sufficient reduction in impurity. Example: min_impurity_decrease': 0.0 (default).
min_impurity_split: It specifies the minimum impurity required for a node to split. This hyperparameter is deprecated in newer versions of scikit-learn, and min_impurity_decrease is preferred.
"""

def decision_tree(train_df, test_df, target_column, hyperparameters=None):
    # Default hyperparameters
    default_hyperparams = {
        'max_depth': 3,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'max_leaf_nodes': 10,
        'min_impurity_decrease': 0.0,
    }

    # If hyperparameters are provided, update the default hyperparameters
    if hyperparameters is not None:
        default_hyperparams.update(hyperparameters)

    # Print hyperparameter values
    print("HYPERPARAMETER VALUES")
    for param, value in default_hyperparams.items():
        print(f"{param}: {value}")

    """
    Train a Decision Tree model with optional hyperparameter tuning and evaluate it on test data.

    :param train_df: Training DataFrame.
    :param test_df: Testing DataFrame.
    :param target_column: The name of the target column.
    :param hyperparameters: Dictionary of hyperparameters for the Decision Tree model.
    :return: None, prints and plots model evaluation metrics.
    """
    # Splitting the train and test data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # Create and fit the Decision Tree model
    model = DecisionTreeClassifier(**default_hyperparams)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Model Accuracy: {accuracy}\n")
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")

    # Plotting Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
# hyperparameters = {'max_depth': 3, 'min_samples_split': 5}
# decision_tree_classification(train_df, test_df, 'target_column_name', hyperparameters)

"""### **Random Forest**"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def random_forest(train_df, test_df, target_column, hyperparameters=None):
    # Default hyperparameters for Random Forest
    default_hyperparams = {
        'n_estimators': 100,  # Default number of trees in the forest
        'max_depth': 3,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'max_leaf_nodes': 10,
        'min_impurity_decrease': 0.0,
    }

    # If hyperparameters are provided, update the default hyperparameters
    if hyperparameters is not None:
        default_hyperparams.update(hyperparameters)

    # Print hyperparameter values
    print("HYPERPARAMETER VALUES")
    for param, value in default_hyperparams.items():
        print(f"{param}: {value}")

    """
    Train a Random Forest model with optional hyperparameter tuning and evaluate it on test data.

    :param train_df: Training DataFrame.
    :param test_df: Testing DataFrame.
    :param target_column: The name of the target column.
    :param hyperparameters: Dictionary of hyperparameters for the Random Forest model.
    :return: None, prints and plots model evaluation metrics.
    """
    # Splitting the train and test data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # Create and fit the Random Forest model
    model = RandomForestClassifier(**default_hyperparams)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Model Accuracy: {accuracy}\n")
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")

    # Plotting Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
# hyperparameters = {'max_depth': 3, 'min_samples_split': 5, 'n_estimators': 200}
# random_forest_classification(train_df, test_df, 'target_column_name', hyperparameters)

"""### **Lightgbm**"""

def lightgbm(train_df, test_df, target_column, val_df=None, hyperparameters=None, optimize_hyperparams=False, optuna_param_grid=None, cv_method=None, cv_folds=5, groups=None, n_repeats=3, p=2, scoring='accuracy'):
    # Splitting the data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    X_val, y_val = None, None
    if val_df is not None:
        X_val = val_df.drop(target_column, axis=1)
        y_val = val_df[target_column]

    # Default hyperparameters for LightGBM
    default_hyperparams = {
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary',
        'metric': 'binary_logloss',
        # Add any additional default parameters you need
    }

    # Update default_hyperparams with given hyperparameters if provided
    if hyperparameters is not None and not optimize_hyperparams:
        default_hyperparams.update(hyperparameters)

    # Optuna hyperparameter optimization
    if optimize_hyperparams:
        # Default parameter grid for Optuna, if not provided
        if optuna_param_grid is None:
            optuna_param_grid = {
                'num_leaves': (20, 40),
                'max_depth': (5, 20),
                'learning_rate': (0.01, 0.2),
                'n_estimators': (50, 200),
                # Add any additional parameters for optimization
            }

        model_type = "lightgbm"
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, model_type, cv_method, cv_folds, groups, n_repeats, p, scoring, optuna_param_grid), n_trials=100)
        # Update hyperparameters after optimization
        default_hyperparams.update(study.best_params)

    # Print hyperparameter values using DataFrame
    print("\nHYPERPARAMETER VALUES:")
    hyper_df = pd.DataFrame.from_dict(default_hyperparams, orient='index', columns=['Value'])
    print(hyper_df)

    # Perform cross-validation if cv_method is specified
    if cv_method is not None:
        cv = get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p)
        model_cv = lgb.LGBMClassifier(**default_hyperparams)
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        print(f"\nCross-Validation Scores ({scoring}): {cv_scores}")
        print(f"Average CV {scoring.capitalize()}: {cv_scores.mean()}")

    # Training the model
    model = lgb.LGBMClassifier(**default_hyperparams)
    if val_df is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # Set verbosity to -1 for silent mode
    else:
        model.fit(X_train, y_train)  # Set verbosity to -1 for silent mode

    # Evaluate the model on train data
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    results_train = calculate_metrics(y_train, y_pred_train, y_proba_train)
    results_train_df = pd.DataFrame.from_dict(results_train, orient='index', columns=['Value'])

    # Evaluate the model on test data
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    results_test = calculate_metrics(y_test, y_pred_test, y_proba_test)
    results_test_df = pd.DataFrame.from_dict(results_test, orient='index', columns=['Value'])

    # Display test data metrics
    #print("\nTEST DATA METRICS:")
    #display_metrics(results_test)

    #Combine results into a single DataFrame
    combined_results = pd.concat([results_train_df, results_test_df], axis=1)
    combined_results.columns = ['Train', 'Test']

    # For colorful display using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_results, annot=True, cmap='magma')
    plt.show()

    display_roc_curve(y_test, y_proba_test)
    display_pr_curve(y_test, y_proba_test)

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    # Print results
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Evaluate the model on validation data if provided
    if val_df is not None:
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        results_val = calculate_metrics(y_val, y_pred_val, y_proba_val)
        results_val_df = pd.DataFrame.from_dict(results_val, orient='index', columns=['Value'])

        #Combine results into a single DataFrame
        combined_results = pd.concat([results_train_df, results_test_df, results_val_df], axis=1)
        combined_results.columns = ['Train', 'Test', 'Validation']

        # For colorful display using seaborn
        plt.figure(figsize=(10, 6))
        sns.heatmap(combined_results, annot=True, cmap='magma')
        plt.show()

        # Display validation data metrics
        #print("\nVALIDATION DATA METRICS:")
        #display_metrics(results_val)

        conf_matrix = confusion_matrix(y_val, y_pred_val)
        class_report = classification_report(y_val, y_pred_val)

        # Print results
        print("Classification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Call the SHAP explainability function
    print("SHAP Analysis:")
    explain_model(model, X_train, X_test, plot_type='bar', num_features=10, plot_individual=True, individual_index=0)

"""### **Catboost**"""

def catboost_help():
    print("CATBOOST_OP FUNCTION USAGE GUIDE\n" + "-"*40)

    print("Function Signature:")
    print("catboost_op(train_df, test_df, target_column, hyperparameters=None, optimize_hyperparams=False, "
          "optuna_param_grid=None, cv_method=None, cv_folds=5, groups=None, n_repeats=3, p=2, scoring='accuracy')\n")

    print("Parameters:")
    print("train_df: DataFrame containing the training data.")
    print("test_df: DataFrame containing the test data.")
    print("target_column: String name of the target column in train_df and test_df.")
    print("hyperparameters: (Optional) Dictionary of hyperparameters for CatBoost. If not provided, default parameters are used.")
    print("optimize_hyperparams: (Optional) Boolean. If True, use Optuna to optimize hyperparameters.")
    print("optuna_param_grid: (Optional) Dictionary of parameter ranges for Optuna optimization. Used if optimize_hyperparams is True.")
    print("cv_method: (Optional) Cross-validation splitting strategy. Can be a string like 'stratified_kfold', 'timeseries', etc.")
    print("cv_folds: (Optional) Number of folds for cross-validation. Default is 5.")
    print("groups: (Optional) Groups to be used in cross-validation.")
    print("n_repeats: (Optional) Number of repeats for RepeatedKFold. Default is 3.")
    print("p: (Optional) Number of samples to leave out in LeavePOut. Default is 2.")
    print("scoring: (Optional) Scoring method for cross-validation. Default is 'accuracy'.\n")

    print("Example Usage:")
    print("# Example DataFrames: train_df, test_df")
    print("# Target column name: 'target_column'")
    print("# Example hyperparameters: {'iterations': 100, 'depth': 4}")
    print("# Optuna parameter grid: {'depth': [4, 6, 8], 'iterations': [100, 200, 300]}")
    print("# To use with default hyperparameters:")
    print("catboost_op(train_df, test_df, 'target_column')")
    print("# To use with custom hyperparameters:")
    print("catboost_op(train_df, test_df, 'target_column', hyperparameters={'iterations': 100, 'depth': 4})")
    print("# To optimize hyperparameters with Optuna:")
    print("catboost_op(train_df, test_df, 'target_column', optimize_hyperparams=True, optuna_param_grid={'depth': [4, 6, 8], 'iterations': [100, 200, 300]})")

    print("\n" + "-"*40)
    print("END OF GUIDE")

def catboost(train_df, test_df, target_column, val_df=None, hyperparameters=None, optimize_hyperparams=False, optuna_param_grid=None, cv_method=None, cv_folds=5, groups=None, n_repeats=3, p=2, scoring='accuracy'):
    # Splitting the data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    X_val, y_val = None, None
    if val_df is not None:
        X_val = val_df.drop(target_column, axis=1)
        y_val = val_df[target_column]

    # Default hyperparameters for CatBoost
    default_hyperparams = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'border_count': 32,
        'loss_function': 'Logloss',
    }

    # Update default_hyperparams with given hyperparameters if provided
    if hyperparameters is not None and not optimize_hyperparams:
        default_hyperparams.update(hyperparameters)

    # Optuna hyperparameter optimization
    if optimize_hyperparams:
        # Default parameter grid for Optuna, if not provided
        if optuna_param_grid is None:
            optuna_param_grid = {
                'iterations': (100, 1000),
                'depth': (3, 10),
                'learning_rate': (0.01, 0.1),
                'l2_leaf_reg': (1, 5),
                'border_count': (5, 50)
            }
        model_type = "catboost"
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, model_type, cv_method, cv_folds, groups, n_repeats, p, scoring, optuna_param_grid), n_trials=100)

    # Print hyperparameter values using DataFrame
    print("\nHYPERPARAMETER VALUES:")
    hyper_df = pd.DataFrame.from_dict(default_hyperparams, orient='index', columns=['Value'])
    print(hyper_df)

    # Perform cross-validation if cv_method is specified
    if cv_method is not None:
        cv = get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p)
        model_cv = CatBoostClassifier(**default_hyperparams, verbose=False)
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        print(f"\nCross-Validation Scores ({scoring}): {cv_scores}")
        print(f"Average CV {scoring.capitalize()}: {cv_scores.mean()}")

    model = CatBoostClassifier(**default_hyperparams, verbose=False)
    if val_df is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    # Evaluate the model on train data
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    results_train = calculate_metrics(y_train, y_pred_train, y_proba_train)
    results_train_df = pd.DataFrame.from_dict(results_train, orient='index', columns=['Value'])

    # Evaluate the model on test data
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    results_test = calculate_metrics(y_test, y_pred_test, y_proba_test)
    results_test_df = pd.DataFrame.from_dict(results_test, orient='index', columns=['Value'])

    # Display test data metrics
    #print("\nTEST DATA METRICS:")
    #display_metrics(results_test)

    #Combine results into a single DataFrame
    combined_results = pd.concat([results_train_df, results_test_df], axis=1)
    combined_results.columns = ['Train', 'Test']

    # For colorful display using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_results, annot=True, cmap='magma')
    plt.show()

    display_roc_curve(y_test, y_proba_test)
    display_pr_curve(y_test, y_proba_test)

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    # Print results
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Evaluate the model on validation data if provided
    if val_df is not None:
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        results_val = calculate_metrics(y_val, y_pred_val, y_proba_val)
        results_val_df = pd.DataFrame.from_dict(results_val, orient='index', columns=['Value'])

        #Combine results into a single DataFrame
        combined_results = pd.concat([results_train_df, results_test_df, results_val_df], axis=1)
        combined_results.columns = ['Train', 'Test', 'Validation']

        # For colorful display using seaborn
        plt.figure(figsize=(10, 6))
        sns.heatmap(combined_results, annot=True, cmap='magma')
        plt.show()

        # Display validation data metrics
        #print("\nVALIDATION DATA METRICS:")
        #display_metrics(results_val)

        conf_matrix = confusion_matrix(y_val, y_pred_val)
        class_report = classification_report(y_val, y_pred_val)

        # Print results
        print("Classification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Call the SHAP explainability function
    print("SHAP Analysis:")
    explain_model(model, X_train, X_test, plot_type='bar', num_features=10, plot_individual=True, individual_index=0)

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import (
    cross_val_score, LeaveOneOut, StratifiedKFold, TimeSeriesSplit, RepeatedKFold,
    LeavePOut, KFold, GroupKFold
)
import seaborn as sns
import matplotlib.pyplot as plt

def catboost_simple(train_df, test_df, target_column, hyperparameters=None, cv_method=None, cv_folds=5, groups=None, n_repeats=3, p=2, scoring='accuracy'):
    # Default hyperparameters for CatBoost
    default_hyperparams = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'border_count': 32,
        'loss_function': 'Logloss',
    }

    if hyperparameters is not None:
        default_hyperparams.update(hyperparameters)

    print("HYPERPARAMETER VALUES")
    for param, value in default_hyperparams.items():
        print(f"{param}: {value}")

    # Splitting the train and test data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # Perform cross-validation if cv_method is specified
    if cv_method is not None:
        # Setting up cross-validator based on the specified method
        cv = get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p)

        # Perform cross-validation
        model_cv = CatBoostClassifier(**default_hyperparams, verbose=False)
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        print(f"\nCross-Validation Scores ({scoring}): {cv_scores}")
        print(f"Average CV {scoring.capitalize()}: {cv_scores.mean()}")

    # Create and fit the CatBoost model on the entire training data
    model = CatBoostClassifier(**default_hyperparams, verbose=False)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"\nModel Accuracy: {accuracy}\n")
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")

    # Plotting Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p):
    if cv_method == 'loocv':
        return LeaveOneOut()
    elif cv_method == 'stratified_kfold':
        return StratifiedKFold(n_splits=cv_folds)
    elif cv_method == 'timeseries':
        return TimeSeriesSplit(n_splits=cv_folds)
    elif cv_method == 'repeated':
        return RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats)
    elif cv_method == 'leave_p_out':
        return LeavePOut(p=p)
    elif cv_method == 'group_kfold':
        if groups is None:
            raise ValueError("Groups must be provided for Group K-Fold CV")
        return GroupKFold(n_splits=cv_folds)
    else:
        return KFold(n_splits=cv_folds)

# Example usage
# hyperparameters = {'depth': 4, 'iterations': 200, 'learning_rate': 0.05}
# catboost_classification_with_optional_cv(train_df, test_df, 'target_column_name', hyperparameters)

"""### **XGboost**"""

def xgboost(train_df, test_df, target_column, val_df=None, hyperparameters=None, optimize_hyperparams=False, optuna_param_grid=None, cv_method=None, cv_folds=5, groups=None, n_repeats=3, p=2, scoring='accuracy'):
    # Splitting the data
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    X_val, y_val = None, None
    if val_df is not None:
        X_val = val_df.drop(target_column, axis=1)
        y_val = val_df[target_column]

    # Default hyperparameters for XGBoost
    default_hyperparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        # Add any additional default parameters you need
    }

    # Update default_hyperparams with given hyperparameters if provided
    if hyperparameters is not None and not optimize_hyperparams:
        default_hyperparams.update(hyperparameters)

    # Optuna hyperparameter optimization
    if optimize_hyperparams:
        # Default parameter grid for Optuna, if not provided
        if optuna_param_grid is None:
            optuna_param_grid = {
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1),
                'colsample_bytree': (0.7, 1),
                # Add any additional parameters for optimization
            }

        model_type = "xgboost"
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, model_type, cv_method, cv_folds, groups, n_repeats, p, scoring, optuna_param_grid), n_trials=100)
        # Update hyperparameters after optimization
        default_hyperparams.update(study.best_params)

    # Print hyperparameter values using DataFrame
    print("\nHYPERPARAMETER VALUES:")
    hyper_df = pd.DataFrame.from_dict(default_hyperparams, orient='index', columns=['Value'])
    print(hyper_df)

    # Perform cross-validation if cv_method is specified
    if cv_method is not None:
        cv = get_cv_strategy(cv_method, cv_folds, groups, n_repeats, p)
        model_cv = XGBClassifier(**default_hyperparams, use_label_encoder=False, eval_metric='logloss')
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        print(f"\nCross-Validation Scores ({scoring}): {cv_scores}")
        print(f"Average CV {scoring.capitalize()}: {cv_scores.mean()}")

    # Training the model
    model = XGBClassifier(**default_hyperparams, use_label_encoder=False, eval_metric='logloss')
    if val_df is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    # Evaluate the model on train data
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    results_train = calculate_metrics(y_train, y_pred_train, y_proba_train)
    results_train_df = pd.DataFrame.from_dict(results_train, orient='index', columns=['Value'])

    # Evaluate the model on test data
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    results_test = calculate_metrics(y_test, y_pred_test, y_proba_test)
    results_test_df = pd.DataFrame.from_dict(results_test, orient='index', columns=['Value'])

    # Display test data metrics
    #print("\nTEST DATA METRICS:")
    #display_metrics(results_test)

    #Combine results into a single DataFrame
    combined_results = pd.concat([results_train_df, results_test_df], axis=1)
    combined_results.columns = ['Train', 'Test']

    # For colorful display using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_results, annot=True, cmap='magma')
    plt.show()

    display_roc_curve(y_test, y_proba_test)
    display_pr_curve(y_test, y_proba_test)

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    # Print results
    print("Classification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Evaluate the model on validation data if provided
    if val_df is not None:
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        results_val = calculate_metrics(y_val, y_pred_val, y_proba_val)
        results_val_df = pd.DataFrame.from_dict(results_val, orient='index', columns=['Value'])

        #Combine results into a single DataFrame
        combined_results = pd.concat([results_train_df, results_test_df, results_val_df], axis=1)
        combined_results.columns = ['Train', 'Test', 'Validation']

        # For colorful display using seaborn
        plt.figure(figsize=(10, 6))
        sns.heatmap(combined_results, annot=True, cmap='magma')
        plt.show()

        # Display validation data metrics
        #print("\nVALIDATION DATA METRICS:")
        #display_metrics(results_val)

        conf_matrix = confusion_matrix(y_val, y_pred_val)
        class_report = classification_report(y_val, y_pred_val)

        # Print results
        print("Classification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Call the SHAP explainability function
    print("SHAP Analysis:")
    explain_model(model, X_train, X_test, plot_type='bar', num_features=10, plot_individual=True, individual_index=0)

"""### **Neural Network**"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

def neural_network(train_df, test_df, target_column, val_df=None, hyperparameters=None, epochs=10, batch_size=32):
    # Preparing the data
    X_train = torch.tensor(train_df.drop(target_column, axis=1).values).float()
    y_train = torch.tensor(train_df[target_column].values).float()
    X_test = torch.tensor(test_df.drop(target_column, axis=1).values).float()
    y_test = torch.tensor(test_df[target_column].values).float()

    # Creating DataLoader for train and test
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Validation data
    val_loader = None
    if val_df is not None:
        X_val = torch.tensor(val_df.drop(target_column, axis=1).values).float()
        y_val = torch.tensor(val_df[target_column].values).float()
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Neural Network Model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, dropout_rate, **kwargs):
            super(NeuralNet, self).__init__()
            layers = []
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                input_size = hidden_size
            layers.append(nn.Linear(hidden_layers[-1], output_size))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Default hyperparameters for Neural Network
    default_hyperparams = {
        'input_size': X_train.shape[1],
        'hidden_layers': [64, 64],
        'output_size': 1,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'activation_function': 'relu',
        'optimizer': 'adam'
        # Add any additional default parameters you need
    }

    # Update default_hyperparams with given hyperparameters if provided
    if hyperparameters is not None:
        default_hyperparams.update(hyperparameters)

    # Initialize the model
    model = NeuralNet(**default_hyperparams)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=default_hyperparams['learning_rate'])

    # Training the model with live loss plotting
    train_loss_values, test_loss_values, val_loss_values = [], [], []

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0.0
        for inputs, labels in train_loader:
            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item() * inputs.size(0)

        # Average training loss
        train_epoch_loss /= len(train_loader.dataset)
        train_loss_values.append(train_epoch_loss)

        # Evaluate test loss
        model.eval()
        test_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                test_epoch_loss += loss.item() * inputs.size(0)

        # Calculate average test loss for the epoch
        test_epoch_loss /= len(test_loader.dataset)
        test_loss_values.append(test_epoch_loss)

        # Evaluate validation loss, if validation data is provided
        if val_df is not None:
            val_epoch_loss = 0.0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_epoch_loss += loss.item() * inputs.size(0)

            val_epoch_loss /= len(val_loader.dataset)
            val_loss_values.append(val_epoch_loss)

        # Live plotting every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:  # Check if it's a multiple of 50 or the last epoch
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_values, label='Training Loss')
            plt.plot(test_loss_values, label='Test Loss')
            if val_df is not None:
                plt.plot(val_loss_values, label='Validation Loss')
            plt.title(f'Training, Test, and Validation Loss (Epoch {epoch+1}/{epochs})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.pause(0.1)
            plt.show()

    # Evaluate the model - Modified to collect predictions and labels
    def evaluate_model(loader, collect_metrics=False):
        model.eval()  # Set the model to evaluation mode
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                if collect_metrics:
                    all_labels.extend(labels.numpy())
                    all_predictions.extend(predicted.numpy())
            if collect_metrics:
                return all_labels, all_predictions
            else:
                accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_predictions))
                return accuracy

    # Function to display metrics and confusion matrix
    def display_metrics_and_confusion_matrix(y_true, y_pred, title):
        print(f"\n{title} Classification Report:")
        print(classification_report(y_true, y_pred))

        conf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.title(f'{title} Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    # Collecting metrics for train, test (and validation if available)
    train_labels, train_predictions = evaluate_model(train_loader, collect_metrics=True)
    test_labels, test_predictions = evaluate_model(test_loader, collect_metrics=True)

    # Display metrics and confusion matrices
    display_metrics_and_confusion_matrix(train_labels, train_predictions, "Train")
    display_metrics_and_confusion_matrix(test_labels, test_predictions, "Test")

    if val_loader is not None:
        val_labels, val_predictions = evaluate_model(val_loader, collect_metrics=True)
        display_metrics_and_confusion_matrix(val_labels, val_predictions, "Validation")

    return model

"""# **Practice**:

### KULLANILABİLECEK KOMUTLAR

* **file_to_df()** -- Her çeşitte dosyayı okuyup dataframe'e dönüştürür.
* **boxplot()** -- Outlier bulmak için tüm scaler verilerin boxplotunu çıkarır.
* **eda()** -- Temel eda fonksiyonlarını gerçekleştirir.
* **detect_outliers()** -- Outlier olan değerleri bulur.
* **detect_outlier_inlist()** -- Listede iletilen kolonlardaki outlier olan değerleri bulur.
* **list_outliers**
* **remove_outliers**
* **histogram_qq()** -- Histogram ve QQ grafiklerini oluşturur.
* **handle_null_values(df, columns, method='drop', fill_value=None, statistic=None)**
* **detect_multicollinearity(df_train, threshold=5.0)**

"""