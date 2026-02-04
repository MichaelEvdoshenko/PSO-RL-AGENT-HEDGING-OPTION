import numpy as np
import pandas as pd
from scipy.stats import norm

def simulate_hekston_paths(params_stock, T_max, N=252, n_paths=1000):
    '''
    params_stock : dict
        Параметры модели: S0, v0, kappa, theta, xi, cor, r, q
    '''
    S0 = params_stock['S0']
    v0 = params_stock['v0']
    kappa = params_stock['kappa']
    theta = params_stock['theta']
    xi = params_stock['xi']
    cor = params_stock['cor']
    r = params_stock['r']
    q = params_stock['q']

    dt = T_max/N
    sqrt_dt = np.sqrt(dt)

    S_all = np.zeros((n_paths, N + 1))
    v_all = np.zeros((n_paths, N + 1))

    S_all[:, 0] = S0
    v_all[:, 0] = v0
    
    Z1 = np.random.normal(0, 1, (n_paths, N))
    Z2 = np.random.normal(0, 1, (n_paths, N))
    
    W_S = Z1

    W_v = cor * Z1 + np.sqrt(1 - cor**2) * Z2

    for t in range(N):
        S_t = S_all[:, t]
        v_t = v_all[:, t]
        
        sqrt_v = np.sqrt(np.maximum(v_t, 0))

        dv = kappa * (theta - v_t) * dt + xi * sqrt_v * sqrt_dt * W_v[:, t]
        v_next = v_t + dv
        v_all[:, t+1] = np.maximum(v_next, 0)

        d_log_S = (r - q - 0.5 * v_t) * dt + sqrt_v * sqrt_dt * W_S[:, t]
        S_all[:, t+1] = S_t * np.exp(d_log_S)
    
    return S_all, v_all



def compute_price_call(params_stock, params_option, N=500, n_paths=1000):

    ''' 
    params_stock : dict
        Параметры модели: v0, kappa, theta, xi, cor
    params_option: pd.DataFrame
        Параметры опциона: [S0, T, K, r, q]
    N : int
        Количество шагов
    n_paths: int
        Количество симуляций
    '''
    T_max = np.max(params_option["T"])

    all_params = params_stock.copy()
    #const parametres of stock
    S0 = params_option["S0"].iloc[0]
    r = params_option["r"].iloc[0]
    q = params_option["q"].iloc[0]
    all_params["S0"] = S0
    all_params["r"] = r
    all_params["q"] = q

    table_of_S, table_of_v = simulate_hekston_paths(all_params, T_max, N, n_paths)
    dt = T_max/N 
    T_all = params_option["T"]

    res_S_0 = []
    T_all_count_of_col = T_all//dt
    for i in range(len(T_all_count_of_col)):
        index = int(T_all_count_of_col.iloc[i])
        v_T = np.mean(table_of_v[index])
        
        K = params_option.iloc[i]["K"]
        payoff = np.mean(np.maximum(table_of_S[:, index] - K, 0))
        S_0 = np.exp(-r*T_all[i]) * payoff
        res_S_0.append(S_0)

    return res_S_0