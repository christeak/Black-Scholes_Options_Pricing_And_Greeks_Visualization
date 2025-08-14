import numpy as np
from scipy.stats import norm

def black_scholes_price(option_type: str, S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calcule le prix d'une option européenne call ou put via le modèle de Black-Scholes.
    
    option_type: 'call' ou 'put'
    S: prix spot de l'actif sous-jacent
    K: prix d'exercice
    r: taux sans risque (en décimal, ex: 0.01)
    sigma: volatilité implicite (en décimal)
    T: maturité en années
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


