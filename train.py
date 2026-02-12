import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import generate_training_data10k as gen
from network import NeuralNetworkPyTorch

import numpy as np
import matplotlib.pyplot as plt


def train():
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim = gen.prepare_data()

    model = NeuralNetworkPyTorch(input_dim = input_dim, hidden_dims = [128, 128, 128])
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # seperate training data in shuffled batches of 64 elements
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )

    # validation data must not be shuffled (order is irrelevant)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),  
        lr = learning_rate          
    )

    train_losses = []
    val_losses = []


    """
    Training phase
    """
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)


        """
        Evaluation Phase
        """
        model.eval()  
        val_loss = 0.0
        
        with torch.no_grad():  
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)


        # Logging (every 10 epochs)
        if (epoch + 1) % 10 == 0:  
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), 'option_pricing_model.pth')
    
    return model, train_losses, val_losses, scaler


def evaluate(model, X_test, y_test, scaler=None):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)

    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test RMSE: ${np.sqrt(test_loss.item()):.4f}")


    # Convert to numpy for analysis
    predictions_np = predictions.numpy().flatten()  
    y_test_np = y_test.numpy().flatten()            
    
    
    # MSE
    mse = np.mean((predictions_np - y_test_np)**2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(predictions_np - y_test_np))
    
    # R² score
    ss_res = np.sum((y_test_np - predictions_np)**2)  
    ss_tot = np.sum((y_test_np - np.mean(y_test_np))**2)  
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE (Mean Absolute Percentage Error)
    # with a mask to avoid "penny" options causing an increase in the percentage
    mask = y_test_np > 1.0
    # mapa = mean(|predicted - actual| / actual * 100)
    mape = np.mean(np.abs((y_test_np[mask] - predictions_np[mask]) / y_test_np[mask]) * 100)   
    
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Masked MAPE: {mape:.2f}%")

    plot(y_test_np = y_test_np, predictions_np = predictions_np)

    return predictions_np, y_test_np
    

"""
Graphs
"""
def plot(y_test_np, predictions_np):
    # Plot 1: Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, predictions_np, alpha=0.5, s=20)
    plt.plot([y_test_np.min(), y_test_np.max()], 
            [y_test_np.min(), y_test_np.max()], 
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Predicted vs Actual Option Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Residual plot
    residuals = predictions_np - y_test_np
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_np, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residual (Predicted - Actual) ($)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('residuals.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Error distribution
    errors = predictions_np - y_test_np
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Error (Predicted - Actual) ($)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. ERROR ANALYSIS
    


if __name__ == "__main__":
    model, _, _, _ = train()
    _, _, X_test, _, _, y_test, scaler, _ = gen.prepare_data()

    predictions, actuals = evaluate(model, X_test, y_test, scaler)

