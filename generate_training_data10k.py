"""
Data Preparation for Neural Network Training
Loads synthetic option data and prepares features for ML model
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define which features to use for training
FEATURE_COLUMNS = [
    # Raw parameters
    'S',                
    'K',                
    'T',                
    'r',                
    'sigma',            
    
    # derived features 
    'moneyness',        
    'sigma_sqrt_T',     
    'intrinsic_value',  
    'd1',               
    'd2'                
]

TARGET_COLUMN = 'price'

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

def load_data(filepath='synthetic_option_data_10k.csv'):
    """
    Load synthetic option data from CSV.
    """
    
    df = pd.read_csv(filepath)
    return df


def extract_features(df, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN):
    """
    Extract feature matrix X and target vector y.
    """
    
    # Check that all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    
    # Check for NaN or inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    if nan_count > 0:
        print(f"\n⚠ WARNING: {nan_count} NaN values in X")
    if inf_count > 0:
        print(f"\n⚠ WARNING: {inf_count} Inf values in X")
        
    return X, y



def split_data(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, 
               test_ratio=TEST_RATIO, random_seed=RANDOM_SEED):
    """
    Split data into train, validation, and test sets.
    X: Feature matrix | y: Target vector
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_ratio, 
        random_state=random_seed
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_seed
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({train_ratio*100:.0f}%)")
    print(f"Val set:   {X_val.shape[0]} samples ({val_ratio*100:.0f}%)")
    print(f"Test set:  {X_test.shape[0]} samples ({test_ratio*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test):
    """
    Normalize features
    """
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Verify normalization on training set
    # print(f"  Mean: {X_train_norm.mean(axis=0)[:3]} ... (should be ~0)")
    # print(f"  Std:  {X_train_norm.std(axis=0)[:3]} ... (should be ~1)")
        
    return X_train_norm, X_val_norm, X_test_norm, scaler


def to_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Convert numpy arrays to PyTorch tensors.
    """
    
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)
    
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t



def prepare_data(filepath='synthetic_option_data_10k.csv', 
                 feature_columns=FEATURE_COLUMNS,
                 return_numpy=False):

    df = load_data(filepath)
    X, y = extract_features(df, feature_columns=feature_columns)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    input_dim = X_train.shape[1]
    
    if return_numpy:
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim
    
    X_train, X_val, X_test, y_train, y_val, y_test = to_tensors(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
        
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim


if __name__ == "__main__":
    # Run the complete pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim = prepare_data()
    
    # Print summary
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nData types:")
    print(f"  X_train: {type(X_train)}")
    print(f"  y_train: {type(y_train)}")