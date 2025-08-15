import numpy as np
from scipy.stats import norm

def black_scholes_price(option_type: str, S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute the price of a European option call or put via the Black-Scholes model.
    
    option_type: 'call' or 'put'
    S: spot price of the underlying asset
    K: strike
    r: interest rate 
    sigma: volatility
    T: maturity in years
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


