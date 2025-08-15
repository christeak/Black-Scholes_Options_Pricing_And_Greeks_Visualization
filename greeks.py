import numpy as np
from scipy.stats import norm

def compute_greeks(option_type: str, S: float, K: float, r: float, sigma: float, T: float):
    """
    Compute the Delta, the Gamma et the Vega of a call option.
    
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega
    }


