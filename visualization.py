import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pricing import black_scholes_price
from greeks import delta_call
from greeks import delta_put
from greeks import gamma
from greeks import vega


# --- Visualization of the payoff ---
def plot_payoff(S_range, K, option_type="call"):
    payoff = []
    for S in S_range:
        if option_type == "call":
            payoff.append(max(S - K, 0))
        elif option_type == "put":
            payoff.append(max(K - S, 0))
    plt.plot(S_range, payoff, label=f"Payoff {option_type}")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Price of the underlying asset at maturity (S_T)")
    plt.ylabel("Payoff (€)")
    plt.title(f"Payoff {option_type.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_delta(S_range, K, r, sigma, T, option_type="call"):
    deltas = []
    for S in S_range:
        if option_type == "call":
            deltas.append(delta_call(S, K, T, r, sigma))
        elif option_type == "put":
            deltas.append(delta_put(S, K, T, r, sigma))

    plt.plot(S_range, deltas, label=f"Delta {option_type.capitalize()}")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(K, color="red", linestyle="--", label="Strike")
    plt.xlabel("Price of the underlying asset")
    plt.ylabel("Delta")
    plt.title(f"Delta {option_type.capitalize()} - Black-Scholes")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_gamma(S_range, K, r, sigma, T):
    gammas = []
    for S in S_range:
        gammas.append(gamma(S, K, T, r, sigma))


    plt.plot(S_range, gammas)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(K, color="red", linestyle="--", label="Strike")
    plt.xlabel("Price of the underlying asset")
    plt.ylabel("Gamma")
    plt.title("Gamma - Black-Scholes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_vega(S, K, r, sigma_range, T):
    vegas = []
    for sigma in sigma_range:
        vegas.append(vega(S, K, T, r, sigma))


    plt.plot(sigma_range, vegas)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Volatility")
    plt.ylabel("Vega")
    plt.title("Vega - Black-Scholes")
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Exemple of use ---
S0 = float(input("Enter the price of the initial underlying S0 : "))   # Prix initial
K = float(input("Enter the strike K : "))                 # Strike
T = float(input("Enter the maturity T : "))                        # Temps (1 an)
r = float(input("Entrez the interest rate r : "))                  # Taux sans risque
sigma = float(input("Enter the volatility sigma : "))                   # Volatilité

print("Price of the call :", black_scholes_price('call', S0, K, r, sigma, T))
print("Price of the put  :", black_scholes_price('put', S0, K, r, sigma, T))
print("Delta (call) :", delta_call(S0, K, T, r, sigma))
print("Delta (put) :", delta_put(S0, K, T, r, sigma))
print("Gamma        :", gamma(S0, K, T, r, sigma))
print("Vega         :", vega(S0, K, T, r, sigma))

S_range = np.linspace(50, 150, 100)
sigma_range = np.linspace(0.01, 1, 100)
plot_payoff(S_range, K, option_type="call")
plot_payoff(S_range, K, option_type="put")
plot_delta(S_range, K, r, sigma, T, "call")
plot_delta(S_range, K, r, sigma, T, "put")
plot_gamma(S_range, K, r, sigma, T)
plot_vega(S0, K, r, sigma_range, T)

