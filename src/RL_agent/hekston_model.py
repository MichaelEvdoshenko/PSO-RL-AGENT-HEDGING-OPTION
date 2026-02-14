import numpy as np
import pandas as pd
import random
from scipy.stats import norm

def compute_price_call_single(S, K, T, r, q, params, n_paths=5000, seed=None):

    v0, kappa, theta, xi, cor = params

    if seed is not None:
        np.random.seed(seed)

    N_steps = max(10, int(T * 252))
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)
    
    S_paths = np.zeros((n_paths, N_steps + 1))
    v_paths = np.zeros((n_paths, N_steps + 1))
    
    S_paths[:, 0] = S
    v_paths[:, 0] = v0

    Z1 = np.random.normal(0, 1, (n_paths, N_steps))
    Z2 = np.random.normal(0, 1, (n_paths, N_steps))

    W_S = Z1
    W_v = cor * Z1 + np.sqrt(1 - cor**2) * Z2
    
    for t in range(N_steps):
        S_t = S_paths[:, t]
        v_t = v_paths[:, t]
        
        sqrt_v = np.sqrt(np.maximum(v_t, 1e-8))

        dv = kappa * (theta - v_t) * dt + xi * sqrt_v * sqrt_dt * W_v[:, t]
        v_next = v_t + dv
        v_paths[:, t+1] = np.maximum(v_next, 1e-8)

        d_log_S = (r - q - 0.5 * v_t) * dt + sqrt_v * sqrt_dt * W_S[:, t]
        S_paths[:, t+1] = S_t * np.exp(d_log_S)
    
    S_T = S_paths[:, -1]
    payoffs = np.maximum(S_T - K, 0)

    price = np.exp(-r * T) * np.mean(payoffs)

    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    np.random.seed(None)
    return price

def calculate_heston_delta(S, K, T, r, q, params):
    """
    params: dict/list
        v0, kappa, theta, xi, cor
    """
    epsilon = 1e-4
    sup = []
    delta = 0.0
    seed = [100, 121, 132, 142, 155, 34, 53, 23, 45, 34]
    for s in seed:
        price_current = compute_price_call_single(S, K, T, r, q, params, seed=s)
        price_up = compute_price_call_single(S + epsilon, K, T, r, q, params, seed=s)
        delta += (price_up - price_current) / epsilon
    delta /=len(seed)
    return delta
    
def calculate_heston_gamma(S, K, T, r, q, params):
    """
    Расчет гаммы (вторая производная)
    """
    epsilon = 1e-4
    
    delta_current = calculate_heston_delta(S, K, T, r, q, params)
    delta_up = calculate_heston_delta(S + epsilon, K, T, r, q, params)
    
    gamma = (delta_up - delta_current) / epsilon
    return gamma