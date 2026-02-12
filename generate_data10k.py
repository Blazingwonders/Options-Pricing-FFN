"""
Synthetic Options Data Generator
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path



def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes implementation.
    """
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if sigma <= 0:
        if option_type == 'call':
            return max(S * np.exp(-r * T) - K, 0)
        else:
            return max(K - S * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def sample_stock_price():
    """
    Sample stock prices.
    -> log-normal distribution centered around $100.
    """
    # Log-normal with mean=$100, std=$30
    S = np.exp(np.random.normal(np.log(100), 0.25))
    return np.clip(S, 20, 500)  # Keep in reasonable range


def sample_strike_price(S):
    """
    Sample strike price relative to stock price.
    Mostly ATM options (see visualisations of synthetic data for more information).
    """
    # Moneyness levels (K/S ratios)
    moneyness_levels = [0.80, 0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15, 1.20]
    probabilities = [0.03, 0.05, 0.10, 0.15, 0.12, 0.20, 0.12, 0.10, 0.07, 0.04, 0.02]
    
    moneyness = np.random.choice(moneyness_levels, p=probabilities)
    K = S * moneyness
    
    K = np.round(K / 5) * 5
    
    return max(K, 5)  # Minimum strike of $5


def sample_time_to_expiry():
    """
    Sample time to expiration.
    Weighted towards shorter maturities.
    """
    # Standard expiration periods in days
    expiry_days = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
    probabilities = [0.12, 0.12, 0.10, 0.18, 0.12, 0.10, 0.10, 0.06, 0.05, 0.03, 0.02]
    
    days = np.random.choice(expiry_days, p=probabilities)
    T = days / 365.0
    
    return T


def sample_risk_free_rate():
    """
    Sample risk-free rate.
    """
    # Centered around 4-5%
    r = np.random.uniform(0.03, 0.06)
    return r


def sample_volatility():
    """
    Sample volatility with regime structure.
    """
    regime = np.random.choice(['low', 'normal', 'high'], p=[0.25, 0.55, 0.20])
    
    if regime == 'low':
        sigma = np.random.uniform(0.10, 0.20)  # Low vol environment
    elif regime == 'normal':
        sigma = np.random.uniform(0.20, 0.35)  # Normal markets
    else:  # high
        sigma = np.random.uniform(0.35, 0.60)  # Crisis/meme stock volatility
    
    return sigma


def create_features(S, K, T, r, sigma):
    """
    The dataset generates a set of 21 columns, which are used to validate 
    the synthetic data. However, the training process (as defined in generate_training_data10k.py) 
    shrinks this down to 10 core features to optimize model performance
    """
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        T = max(T, 1e-6)
        sigma = max(sigma, 1e-6)
    
    # Basic features
    moneyness = S / K
    log_moneyness = np.log(S / K)
    
    # Time-scaled features
    sqrt_T = np.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    
    # Intrinsic value
    intrinsic_value = max(S - K, 0)  # For calls
    
    # Black-Scholes d1 and d2 (give model a hint)
    d1 = (log_moneyness + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    
    # Time value factor
    discount_factor = np.exp(-r * T)
    
    # Forward price
    forward_price = S * np.exp(r * T)
    forward_moneyness = forward_price / K
    
    features = {
        # Raw parameters
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        
        # Derived features
        'moneyness': moneyness,
        'log_moneyness': log_moneyness,
        'sqrt_T': sqrt_T,
        'sigma_sqrt_T': sigma_sqrt_T,
        'intrinsic_value': intrinsic_value,
        'd1': d1,
        'd2': d2,
        'discount_factor': discount_factor,
        'forward_price': forward_price,
        'forward_moneyness': forward_moneyness,
    }
    
    return features


def enforce_arbitrage_bounds(price, S, K, T, r, option_type='call'):
    """
    Ensure option price satisfies no-arbitrage bounds.
    - Lower bound: max(S - K*e^(-rT), 0)
    - Upper bound: S
    """
    if option_type == 'call':
        lower_bound = max(S - K * np.exp(-r * T), 0)
        upper_bound = S
        
        if price < lower_bound:
            price = lower_bound + 0.001  
        if price > upper_bound:
            price = upper_bound * 0.999
    
    # Ensure positive
    price = max(price, 0.001)
    
    return price


def validate_sample(S, K, T, r, sigma, price):
    """
    Run quality checks on a single sample.
    Returns True if valid, False otherwise.
    """
    # Check for NaN or inf
    if not np.isfinite([S, K, T, r, sigma, price]).all():
        return False
    
    # Check positive values
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or price < 0:
        return False
    
    """
    Check reasonable ranges
    S and K <= 1000; T <= 5; volatility <= 200%; r <= 20%
    """

    if S > 1000 or K > 1000:  
        return False
    if T > 5:  # More than 5 years
        return False
    if sigma > 2.0:  # 200% volatility is extreme
        return False
    if r > 0.20:  # 20% risk-free rate is unrealistic
        return False
    
    # Check arbitrage bounds for calls
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if price < intrinsic * 0.99:  # Allow small tolerance
        return False
    if price > S * 1.01:
        return False
    
    return True


def generate_synthetic_data(num_samples, add_noise=True, noise_level=0.01, random_seed=42):
    """
    Generate synthetic option pricing data.
    """
    np.random.seed(random_seed)
    
    data = []
    samples_generated = 0
    attempts = 0
    max_attempts = num_samples * 10 
    
    while samples_generated < num_samples and attempts < max_attempts:
        attempts += 1
        
        # 1. Sample base parameters
        S = sample_stock_price()
        K = sample_strike_price(S)
        T = sample_time_to_expiry()
        r = sample_risk_free_rate()
        sigma = sample_volatility()
        
        # 2. Calculate Black-Scholes price 
        price_bs = black_scholes(S, K, T, r, sigma, 'call')
        
        # 3. Add noise (noise level is set to 1% in default)
        if add_noise:
            # Spread increases for away-from-money options
            moneyness = S / K
            distance_from_atm = abs(np.log(moneyness))
            
            # Base spread + liquidity premium
            spread_pct = noise_level + 0.01 * distance_from_atm
            spread_pct = min(spread_pct, 0.05)  # Cap at 5%
            
            noise = np.random.uniform(-spread_pct, spread_pct) * price_bs
            price_market = price_bs + noise
        else:
            price_market = price_bs
        
        # 4. Enforce no-arbitrage bounds
        price_market = enforce_arbitrage_bounds(price_market, S, K, T, r, 'call')
        
        # 5. Validate sample
        if not validate_sample(S, K, T, r, sigma, price_market):
            continue
        
        # 6. Create features
        features = create_features(S, K, T, r, sigma)
        
        # 7. Store all information
        sample = {
            # Raw parameters
            'S': S,
            'K': K,
            'T': T,
            'T_days': T * 365,
            'r': r,
            'sigma': sigma,
            
            # Target
            'price': price_market,
            'price_bs': price_bs,  # For comparison
            
            # Derived features
            **features,
            
            # Metadata
            'moneyness_category': categorize_moneyness(S / K),
            'time_category': categorize_time(T),
            'vol_category': categorize_volatility(sigma),
        }
        
        data.append(sample)
        samples_generated += 1
        
        # Progress update
        if samples_generated % 1000 == 0:
            print(f"Generated {samples_generated}/{num_samples} samples...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df


def categorize_moneyness(moneyness):
    """Categorize option by moneyness."""
    if moneyness < 0.95:
        return 'OTM'
    elif moneyness > 1.05:
        return 'ITM'
    else:
        return 'ATM'


def categorize_time(T):
    """Categorize option by time to expiration."""
    if T < 30/365:
        return 'short'
    elif T < 90/365:
        return 'medium'
    else:
        return 'long'


def categorize_volatility(sigma):
    """Categorize by volatility regime."""
    if sigma < 0.20:
        return 'low'
    elif sigma < 0.35:
        return 'normal'
    else:
        return 'high'


# Data visualization and further validation

def validate_dataset(df):
    """
    dataset validation with visualizations.
    """
    
    # Basic statistics
    print("\nDataset Shape:", df.shape)
    print("\nBasic Statistics:")
    print(df[['S', 'K', 'T', 'r', 'sigma', 'price']].describe())
    
    
    # Price vs moneyness (should be decreasing as K increases)
    corr = df['price'].corr(df['moneyness'])
    print(f"Price vs Moneyness correlation: {corr:.4f}")
    
    # Price vs time
    corr = df['price'].corr(df['T'])
    print(f"Price vs Time correlation: {corr:.4f}")
    
    # Price vs volatility 
    corr = df['price'].corr(df['sigma'])
    print(f"Price vs Volatility correlation: {corr:.4f}")
    
    # Check arbitrage bounds
    intrinsic = np.maximum(df['S'] - df['K'] * np.exp(-df['r'] * df['T']), 0)
    arbitrage_violations = (df['price'] < intrinsic * 0.99).sum()
    print(f"\nArbitrage violations: {arbitrage_violations}")
    
    # Check price > 0
    negative_prices = (df['price'] <= 0).sum()
    print(f"Negative prices: {negative_prices}")
    
    return df


def plot_data_analysis(df, save_path='data_analysis.png'):
    """
    Create visualizations
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Synthetic Option Data Analysis', fontsize=16, y=1.00)
    
    # 1. Price distribution
    axes[0, 0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Option Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Price vs Moneyness
    axes[0, 1].scatter(df['moneyness'], df['price'], alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Moneyness (S/K)')
    axes[0, 1].set_ylabel('Option Price ($)')
    axes[0, 1].set_title('Price vs Moneyness')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
    axes[0, 1].legend()
    
    # 3. Price vs Time to Expiry
    axes[0, 2].scatter(df['T_days'], df['price'], alpha=0.3, s=10)
    axes[0, 2].set_xlabel('Time to Expiry (days)')
    axes[0, 2].set_ylabel('Option Price ($)')
    axes[0, 2].set_title('Price vs Time to Expiry')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Price vs Volatility
    axes[1, 0].scatter(df['sigma'], df['price'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Volatility (σ)')
    axes[1, 0].set_ylabel('Option Price ($)')
    axes[1, 0].set_title('Price vs Volatility')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Volatility distribution
    axes[1, 1].hist(df['sigma'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Volatility (σ)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Volatility Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Time to expiry distribution
    axes[1, 2].hist(df['T_days'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Time to Expiry (days)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Time to Expiry Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Stock price distribution
    axes[2, 0].hist(df['S'], bins=40, edgecolor='black', alpha=0.7)
    axes[2, 0].set_xlabel('Stock Price ($)')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Stock Price Distribution')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Moneyness category counts
    moneyness_counts = df['moneyness_category'].value_counts()
    axes[2, 1].bar(moneyness_counts.index, moneyness_counts.values, 
                   edgecolor='black', alpha=0.7)
    axes[2, 1].set_xlabel('Moneyness Category')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_title('Distribution by Moneyness')
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    # 9. Price error
    if 'price_bs' in df.columns:
        price_error = df['price'] - df['price_bs']
        axes[2, 2].hist(price_error, bins=50, edgecolor='black', alpha=0.7)
        axes[2, 2].set_xlabel('Price - BS Price ($)')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title('Market Noise Distribution')
        axes[2, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    # Generate data
    df = generate_synthetic_data(
        num_samples=10000,
        add_noise=True,
        noise_level=0.01,  # 1% base noise
        random_seed=42
    )
    
    # Validate
    df = validate_dataset(df)
    
    # Visualize
    plot_data_analysis(df, save_path='synthetic_data_analysis.png')
    
    # Save to CSV
    output_file = 'synthetic_option_data_10k.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")
        
    
    # Print sample
    # print("SAMPLE DATA (first 5 rows)\n")
    # print(df[['S', 'K', 'T_days', 'r', 'sigma', 'moneyness', 'price']].head())
    
    print("DATA GENERATION COMPLETE!\n")