import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle as pkl
import argparse
import os

def load_and_clean_data(file_path):
    """
    Load the dataset from a CSV file and clean it by dropping missing values.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna()
    return df_cleaned

def encode_categorical_features(df, columns, prefix):
    """
    Apply one-hot encoding to the specified categorical columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list of str): The list of categorical columns to encode.
    - prefix (str): The prefix for the new encoded columns.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    df_encoded = pd.get_dummies(df, columns=columns, prefix=prefix, dtype='int64')
    return df_encoded

def apply_pca(data, target_column, test_size=0.2, random_state=123):
    """
    Apply PCA to the features of the dataset to handle collinearity, while keeping the target variable aside.

    Parameters:
    - data (pd.DataFrame): The input dataset containing features and target variable.
    - target_column (str): The name of the target column.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - train_pca_with_target (pd.DataFrame): The PCA-transformed training data with the target variable.
    - test_pca_with_target (pd.DataFrame): The PCA-transformed test data with the target variable.
    """
    # Separate features and target
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Convert the PCA-transformed data back to DataFrame
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

    # Concatenate the PCA features with the target
    train_pca_with_target = pd.concat([X_train_pca_df, y_train.reset_index(drop=True)], axis=1)
    test_pca_with_target = pd.concat([X_test_pca_df, y_test.reset_index(drop=True)], axis=1)

    return train_pca_with_target, test_pca_with_target

def validation_split(data, target_column, val_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.

    Parameters:
    - data (pd.DataFrame): The input dataset containing features and target variable.
    - target_column (str): The name of the target column.
    - val_size (float): The proportion of the dataset to include in the validation split.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - X_train, X_val, y_train, y_val (pd.DataFrame): The training and validation features and target variables.
    """
    # Separate features and target
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

def preprocess_data(file_path, target_column, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline including data loading, cleaning, encoding, and PCA.

    Parameters:
    - file_path (str): The path to the CSV file.
    - target_column (str): The name of the target column.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - X_train, X_val, y_train, y_val, X_test, y_test (pd.DataFrame): The preprocessed data split into training, validation, and test sets.
    """
    # Load and clean data
    df_cleaned = load_and_clean_data(file_path)

    # Encode categorical features
    df_one_hot_encoded = encode_categorical_features(df_cleaned, columns=['ocean_proximity'], prefix='proximity')

    # Apply PCA and split into training and test sets
    train_df, test_df = apply_pca(df_one_hot_encoded, target_column, test_size=test_size, random_state=random_state)

    # Validation split
    X_train, X_val, y_train, y_val = validation_split(train_df, target_column, val_size=0.15, random_state=432)

    # Separate features and target
    y_test = test_df[target_column]
    X_test = test_df.drop(target_column, axis=1)

    return X_train, X_val, y_train, y_val, X_test, y_test

def save_data(data, file_path):
    """
    Save the processed data to a pickle file.

    Parameters:
    - data (dict): The dictionary containing processed data splits.
    - file_path (str): The path to save the pickle file.
    """
    with open(file_path, "wb") as fout:
        pkl.dump(data, fout)

def main(input_file, output_file, target_column):
    """
    Main function to preprocess the data and save it to a pickle file.

    Parameters:
    - input_file (str): The path to the input CSV file.
    - output_file (str): The path to the output pickle file.
    - target_column (str): The name of the target column.
    """
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_data(input_file, target_column)

    data = {
        "x_train": X_train,
        "x_val": X_val,
        "x_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    save_data(data, output_file)

    print("Preprocessing complete. Data saved to:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and save as a pickle file.")
    parser.add_argument("input_file", type=str, nargs='?', default=os.path.join(os.getcwd(), "housing.csv"), help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, nargs='?', default=os.path.join(os.getcwd(), "preprocessed_housing.pkl"), help="Path to the output pickle file.")
    parser.add_argument("target_column", type=str, help="Name of the target column.")

    args = parser.parse_args()

    main(args.input_file, args.output_file, args.target_column)