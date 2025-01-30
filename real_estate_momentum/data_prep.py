import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sliding_windows(returns_data, raw_data, window_size, prediction_size):
    """
    Create sliding windows for time-series data.

    Args:
        returns_data (pd.DataFrame): DataFrame of returns for a single region (columns=1).
        raw_data (pd.DataFrame): DataFrame of raw (ZHVI) values for a single region (columns=1).
        window_size (int): The length of the historical window used as features.
        prediction_size (int): The forecast horizon.

    Returns:
        (np.array, np.array): X and y arrays.
    """
    X, y = [], []
    for i in range(len(returns_data) - window_size - prediction_size + 1):
        window_returns = returns_data[i:i + window_size].values
        window_raw = raw_data[i:i + window_size].values
        window_combined = np.hstack((window_raw, window_returns))  # shape: (window_size, 2)
        X.append(window_combined)
        y.append(returns_data.iloc[i + window_size + prediction_size - 1].values)  # final return in horizon
    return np.array(X), np.array(y)

def load_and_preprocess_data(
    zhvi_path,
    msa_datamap_path,
    window_size=12,
    prediction_size=12,
    val_date='2016-01-01',
    split_date='2020-01-01'
):
    """
    Loads ZHVI data, creates returns, splits into train/val/test, and scales.
    
    Args:
        zhvi_path (str): Path to the ZHVI CSV file.
        msa_datamap_path (str): Path to the MSA data map CSV file.
        window_size (int): Sliding window size (in months).
        prediction_size (int): Prediction horizon (in months).
        val_date (str): Cutoff date for train/val split.
        split_date (str): Cutoff date for val/test split.

    Returns:
        dict: A dictionary containing:
            - X_train, y_train, X_val, y_val, X_test, y_test (all numpy arrays)
            - scaler (the fitted MinMaxScaler)
            - df_zhvi (original ZHVI DataFrame, indexed by Date)
            - msa_datamap (the DataFrame with region info)
            - device (the PyTorch device in use)
            - num_assets (number of regions)
    """
    import torch  # local import so the file can also be used without PyTorch if needed

    # 1. Load data
    df_zhvi = pd.read_csv(zhvi_path, index_col='Date', parse_dates=True)
    msa_datamap = pd.read_csv(msa_datamap_path)
    
    # 2. Create 1-year returns
    returns_1y = df_zhvi.pct_change(periods=prediction_size)

    # 3. Prepare train sets
    returns_1y_train = returns_1y[returns_1y.index < val_date]
    df_zhvi_train = df_zhvi[df_zhvi.index < val_date]
    
    X_train_list, y_train_list = [], []
    for column in returns_1y.columns:
        region_returns = returns_1y_train[[column]].dropna()
        region_raw = df_zhvi_train[[column]].dropna()[prediction_size:]
        if len(region_returns) >= (window_size + prediction_size):
            region_X, region_y = create_sliding_windows(region_returns, region_raw, window_size, prediction_size)
            X_train_list.append(region_X)
            y_train_list.append(region_y)

    X_train = np.concatenate(X_train_list, axis=0) if len(X_train_list) > 0 else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if len(y_train_list) > 0 else np.array([])

    # 4. Prepare validation sets
    returns_1y_val = returns_1y[(returns_1y.index >= val_date) & (returns_1y.index < split_date)]
    df_zhvi_val = df_zhvi[(df_zhvi.index >= val_date) & (df_zhvi.index < split_date)]
    
    X_val_list, y_val_list = [], []
    for column in returns_1y.columns:
        region_returns = returns_1y_val[[column]].dropna()
        region_raw = df_zhvi_val[[column]].dropna()
        if len(region_returns) >= (window_size + prediction_size):
            region_X, region_y = create_sliding_windows(region_returns, region_raw, window_size, prediction_size)
            X_val_list.append(region_X)
            y_val_list.append(region_y)

    X_val = np.concatenate(X_val_list, axis=0) if len(X_val_list) > 0 else np.array([])
    y_val = np.concatenate(y_val_list, axis=0) if len(y_val_list) > 0 else np.array([])

    # 5. Prepare test sets
    returns_1y_test = returns_1y[returns_1y.index >= split_date]
    df_zhvi_test = df_zhvi[df_zhvi.index >= split_date]

    X_test_list, y_test_list = [], []
    for column in returns_1y.columns:
        region_returns = returns_1y_test[[column]].dropna()
        region_raw = df_zhvi_test[[column]].dropna()
        if len(region_returns) >= (window_size + prediction_size):
            region_X, region_y = create_sliding_windows(region_returns, region_raw, window_size, prediction_size)
            X_test_list.append(region_X)
            y_test_list.append(region_y)

    X_test = np.concatenate(X_test_list, axis=0) if len(X_test_list) > 0 else np.array([])
    y_test = np.concatenate(y_test_list, axis=0) if len(y_test_list) > 0 else np.array([])

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape},   y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

    # 6. Scale data
    # IMPORTANT: We fit only on X_train, then transform X_val and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    if X_train.size > 0:
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    else:
        # Edge case if no data
        X_train_scaled = X_train
        X_val_scaled   = X_val
        X_test_scaled  = X_test

    # 7. Convert y to the correct shape
    y_train = y_train.reshape(-1, 1)
    y_val   = y_val.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)

    # 8. Get device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device in use:", device)

    # 9. Count the number of assets (top-100 if your file has 100 columns)
    num_assets = df_zhvi.shape[1]

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'df_zhvi': df_zhvi,
        'msa_datamap': msa_datamap,
        'device': device,
        'num_assets': num_assets
    }
