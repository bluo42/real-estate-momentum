import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Local imports
from data_prep import load_and_preprocess_data
from dataset import TimeSeriesDataset
from models import LSTMModel
from train import train_model
from evaluation import evaluate, compute_metrics

def main():
    # -------------------------------
    # 1) Load & preprocess the data
    # -------------------------------
    data_dict = load_and_preprocess_data(
        zhvi_path='../data/processed/zhvi_top100_ts.csv',
        msa_datamap_path='../data/processed/msa_datamap.csv',
        window_size=12,
        prediction_size=12,
        val_date='2016-01-01',
        split_date='2020-01-01'
    )

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val   = data_dict['X_val']
    y_val   = data_dict['y_val']
    X_test  = data_dict['X_test']
    y_test  = data_dict['y_test']
    df_zhvi  = data_dict['df_zhvi']
    msa_map  = data_dict['msa_datamap']
    device   = data_dict['device']
    num_assets = data_dict['num_assets']

    # -------------------------------
    # 2) Build PyTorch Datasets
    # -------------------------------
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset   = TimeSeriesDataset(X_val,   y_val)
    test_dataset  = TimeSeriesDataset(X_test,  y_test)

    # -------------------------------
    # 3) Create Dataloaders
    # -------------------------------
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    # -------------------------------
    # 4) Initialize Model
    # -------------------------------
    input_size  = 2   # (raw_value, return_value)
    hidden_size = 50
    num_layers  = 1
    output_size = 1   # single-step forecast
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # -------------------------------
    # 5) Train Model
    # -------------------------------
    num_epochs       = 50
    reporting_epochs = 5

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        device=device,
        num_epochs=num_epochs,
        reporting_epochs=reporting_epochs,
        lr=0.001,
        weight_decay=1e-5
    )

    # Plot the training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    # -------------------------------
    # 6) Evaluate Model
    # -------------------------------
    y_pred = evaluate(model, test_loader, device)
    mse_lstm, mae_lstm = compute_metrics(y_test, y_pred)
    print("LSTM Model - MSE: ", mse_lstm)
    print("LSTM Model - MAE: ", mae_lstm)

    # -------------------------------
    # 7) Benchmark
    #    "Benchmark" = last return from the input window
    #    For each test sample, that means the last time-step's return in the window.
    # -------------------------------
    # Because we processed them as big concatenated arrays, we replicate the approach:
    # The last return in each test window is in X_test[..., 1], specifically the last row is X_test[i, -1, 1]
    # But we must ensure we do this in the same order as test_dataset.
    benchmark_preds = []
    for i in range(X_test.shape[0]):
        # The second column in each time-step is the "return" since input_size=2 -> (raw, ret).
        # The last time-step's return in that window is:
        last_return = X_test[i, -1, 1]
        benchmark_preds.append(last_return)
    benchmark_preds = np.array(benchmark_preds).reshape(-1, 1)

    mse_benchmark, mae_benchmark = compute_metrics(y_test, benchmark_preds)
    print("Benchmark Model - MSE: ", mse_benchmark)
    print("Benchmark Model - MAE: ", mae_benchmark)

    # -------------------------------
    # 8) Create a DataFrame with predictions vs. actual
    #    We assume each region contributed the same number of samples in test. 
    #    If the shape is not a perfect multiple, some logic might differ.
    # -------------------------------
    # For simplicity, let's assume each region has the same # of windows in test. 
    # We'll reconstruct the region indexing by chunking the predictions.
    n_samples_per_asset = len(test_dataset) // num_assets

    # If you want just the "last" prediction for each region, you might do something like:
    #   y_pred[::n_samples_per_asset] (picking every nth to align with region ordering)
    # The original code used y_pred[:: int(y_pred.shape[0]/num_assets)][::-1], etc.
    # We'll replicate something similar here:
    selected_indices = range(0, len(y_pred), n_samples_per_asset)
    region_pred = y_pred[selected_indices].flatten()
    region_actual = y_test[selected_indices].flatten()
    region_bench = benchmark_preds[selected_indices].flatten()

    # Build a DataFrame
    pred_df = []
    for i, col in enumerate(df_zhvi.columns):
        # We assume i matches the chunk:
        pred_df.append((col, region_pred[i], region_actual[i], region_bench[i]))
    pred_df = np.array(pred_df, dtype=object)

    pred_df = {
        'RegionID': pred_df[:, 0],
        'PredictedReturn': pred_df[:, 1],
        'ActualReturn': pred_df[:, 2],
        'BenchmarkReturn': pred_df[:, 3]
    }

    import pandas as pd
    pred_df = pd.DataFrame(pred_df)

    # Convert to numeric if RegionID was numeric
    pred_df['RegionID'] = pred_df['RegionID'].astype(int, errors='ignore')

    # Merge with MSA data
    merged_df = pd.merge(pred_df, msa_map, on='RegionID', how='left')
    sorted_df = merged_df.sort_values(by='PredictedReturn', ascending=False)
    
    print("Top of sorted predictions:")
    print(sorted_df.head())

    # Save to CSV
    sorted_df.to_csv('analysis.csv', index=False)
    print("Saved analysis.csv!")

if __name__ == "__main__":
    main()
