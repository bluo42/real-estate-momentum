import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

def evaluate(model, test_loader, device):
    """
    Run forward pass on the test set to obtain predictions.
    """
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred.append(outputs.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred

def compute_metrics(y_true, y_pred):
    """
    Compute MSE and MAE between predictions and true values.
    """
    mse_val = mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    return mse_val, mae_val
