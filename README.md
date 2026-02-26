# Neural Network Option Pricing
A feedforward neural network that learns to price European call options, achieving 99.97% accuracy (R² = 0.9997) on synthetic Black-Scholes data.

### Architecture
Input: 10 features (S, K, T, r, σ, moneyness, σ√T, intrinsic value, d₁, d₂)
Hidden Layers: 128 → 128 → 128 neurons (ReLU activation)
Output: Single option price
Framework: PyTorch

### Results
Test Set Performance:
- RMSE: $0.10
- MAE: $0.07
- R²: 0.9997

### Installation
```
bashpip install torch numpy pandas scikit-learn matplotlib
```

### Usage
```
1. run generate_training_data10k.py
2. run train.py
3. look at the results and visualizations and tweak the model
```

### Project Structure
```
├── network.py                        # Neural network architecture
├── data_preparation.py               # Data loading and preprocessing
├── train.py                          # Training script
├── generate_training_data10k.py      # Synthetic data generator
├── requirements.txt      
```

### Visualizations
![Picture showing properties of synthetic data](synthetic_data_analysis.png)

Technologies
PyTorch • NumPy • Pandas • Scikit-learn
