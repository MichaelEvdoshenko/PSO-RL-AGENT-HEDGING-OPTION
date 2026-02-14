import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def BSM_compute_price(params, vol):
    '''
    params: dict     
        S0, K, T, r, q
    vol: int
    '''
    S0 = params["S0"]
    r = params["r"]
    q = params["q"]

    K = params["K"]
    T = params["T"]

    d1 = (np.log(S0/K) + (r - q + (vol**2)/2) * T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    C_price = S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    return C_price

def compute_vol_BSM(params, C_price, vol_low=0.0001, vol_high=5.0, eps=1e-8, max_iter=1000):
    '''
    params: dict
        S0, K, T, r, q
    C_price: int
        цена опционов колл
    '''
    
    S0 = params["S0"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    q = params["q"]

    min_price = max(0, S0 * np.exp(-q * T) - K * np.exp(-r * T)) #максимальная теоретическая цена
    max_price = S0 * np.exp(-q * T) #минимальная теоретическая цена
    
    # Проверка арбитражных условий
    if C_price < min_price - 1e-10:
        return np.nan
    if C_price > max_price + 1e-10:
        return np.nan
    
    if abs(C_price - min_price) < eps:
        return vol_low
    
    price_low = BSM_compute_price(params, vol_low)
    price_high = BSM_compute_price(params, vol_high)
    
    if price_low > C_price:
        vol_low = 0.0001
        price_low = BSM_compute_price(params, vol_low)
        if price_low > C_price:
            return np.nan
    
    if price_high < C_price:
        return np.nan
    
    for iteration in range(max_iter):
        vol_middle = (vol_high + vol_low) / 2
        price_middle = BSM_compute_price(params, vol_middle)
        
        if abs(price_middle - C_price) < eps:
            return vol_middle
        
        if ((price_middle - C_price) > 0):
            vol_high = vol_middle
        else:
            vol_low = vol_middle
    
    return (vol_low + vol_high) / 2