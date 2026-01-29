import numpy as np
import matplotlib.pyplot as plt

def compute_price_call(params_stock, params_option, N=252, n_paths=10000):

    ''' 
    params_stock : dict
        Параметры модели: S0, v0, kappa, theta, xi, rho, r, q
    params_option: dict
        Параметры опциона: T, K
    N : int
        Количество шагов
    n_paths: int
        Количество симуляций
    '''
    
    S0 = params_stock['S0']
    v0 = params_stock['v0']
    kappa = params_stock['kappa']
    theta = params_stock['theta']
    xi = params_stock['xi']
    cor = params_stock['cor']
    r = params_stock['r']
    q = params_stock['q']

    T = params_option['T']
    K = params_option['K']

    dt = T/N
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

    S_T = S_all[:, -1]
    call_payoffs = np.maximum(S_T - K, 0)
    
    call_price = np.exp(-r * T) * np.mean(call_payoffs)

    v_T = v_all[:, -1]
    mean_v = np.mean(v_T)
    
    return call_price, mean_v
