import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, device, num_epochs=50, reporting_epochs=5, lr=0.001, weight_decay=1e-5):
    """
    Train an LSTM model on time-series data.

    Args:
        model (nn.Module): The LSTM model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to use for training (GPU or CPU).
        num_epochs (int): Number of epochs to train.
        reporting_epochs (int): Frequency of printing training progress.
        lr (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 regularization).

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_train_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss_val = criterion(val_outputs, y_val)
                val_loss += loss_val.item()
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Print progress
        if (epoch + 1) % reporting_epochs == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Training Loss: {epoch_train_loss:.8f}, "
                  f"Validation Loss: {epoch_val_loss:.8f}")

    return model, train_losses, val_losses
