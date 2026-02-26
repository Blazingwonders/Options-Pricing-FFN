**Neural Network Option Pricing**
A feedforward neural network that learns to price European call options, achieving 99.97% accuracy (R² = 0.9997) on synthetic Black-Scholes data.

# Architecture
Input: 10 features (S, K, T, r, σ, moneyness, σ√T, intrinsic value, d₁, d₂)
Hidden Layers: 128 → 128 → 128 neurons (ReLU activation)
Output: Single option price
Framework: PyTorch

# Results
Test Set Performance:
- RMSE: $0.10
- MAE: $0.07
- R²: 0.9997

# Installation
```
bashpip install torch numpy pandas scikit-learn matplotlib
```

# Usage
```
pythonfrom model import NeuralNetworkPyTorch
from data_preparation import prepare_data
```
# Load data
```
X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim = prepare_data()
```

# Create and train model
```
model = NeuralNetworkPyTorch(input_dim=10, hidden_dims=[128, 128, 128])
```

## Project Structure
```
├── model.py                          # Neural network architecture
├── data_preparation.py               # Data loading and preprocessing
├── train.py                          # Training script
├── generate_training_data10k.py      # Synthetic data generator
└── synthetic_option_data_10k.csv     # Training dataset
```
Technologies
PyTorch • NumPy • Pandas • Scikit-learn
