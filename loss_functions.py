import pandas as pd
import numpy as np
from simulate_hekston import compute_price_call
from BSM import compute_vol_BSM

def squad_error_for_all_options(params_stock, market_data):
    
    """ params_stock: dict/np.ndarray
            вычисленные параметры модели: v0, kappa, theta, xi, cor
        market_data: DataFrame
            рыночные данные опциона и акции ["S0", "T", "K", "r", "q"]
    """

    params_options = market_data[["S0", "T", "K", "r", "q"]].copy()
    if (type(params_stock) == np.ndarray):
            names = ['v0', 'kappa', 'theta', 'xi', 'cor']
            params_stock = dict(zip(names, params_stock))  
    S_0 = compute_price_call(params_stock, params_options)
    total_error = np.sum((S_0 - market_data["call_mid"])**2)
    
    mse = total_error / len(market_data)
    return mse

def heston_loss_function(params_stock, market_data):
    '''
    params_stock: dict/np.ndarray
        вычисленные параметры модели: v0, kappa, theta, xi, cor
    market_data: DataFrame
        рыночные данные опциона и акции ['K', 'T', 'market_iv', 'weight', 'S0', 'r', 'q']
    '''

    v0, kappa, theta, sigma_v, cor = params_stock

    heston_prices = []
    for idx, par in df.iterrows():
        params_options = market_data[["S0", "T", "K", "r", "q"]].copy()
        if (type(params_stock) == np.ndarray):
            names = ['v0', 'kappa', 'theta', 'xi', 'cor']
            params_stock = dict(zip(names, params_stock))
        price = compute_price_call(params_stock, params_options)
        heston_prices.append(price)
    
    model_iv_list = []
    for i, (price, (idx, row)) in enumerate(zip(heston_prices, df.iterrows())):
        iv = compute_vol_BSM(row, price)
        model_iv_list.append(iv)
    
    market_iv = df['market_iv'].values
    model_iv = np.array(model_iv_list)