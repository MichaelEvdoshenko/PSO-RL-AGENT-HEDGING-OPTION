import numpy as np
import matplotlib.pyplot as plt

def simulate_heston_path(params, T=1.0, N=252):

    ''' 
    params : dict
        Параметры модели: S0, v0, kappa, theta, xi, rho, r, q
    T : float
        Время в годах
    N : int
        Количество шагов
    '''
    
    S0 = params['S0']
    v0 = params['v0']
    kappa = params['kappa']
    theta = params['theta']
    xi = params['xi']
    cor = params['cor']
    r = params['r']
    q = params['q']

    dt = T/N

    S_path = np.zeros(N + 1)
    v_path = np.zeros(N + 1)
    S_path[0] = S0
    v_path[0] = v0
    
    for t in range(N):
        
        S_t = S_path[t]
        v_t = v_path[t]

        KSI_1 = np.random.normal(0, 1)  # Для цены
        KSI_2 = np.random.normal(0, 1)  # Для волатильности

        W_S = KSI_1
        W_v = cor * KSI_1 + np.sqrt(1 - cor**2) * KSI_2

        sqrt_dt = np.sqrt(dt)        
        sqrt_v = np.sqrt(max(v_t, 0))
        dv = kappa * (theta - v_t) * dt + xi * sqrt_v * sqrt_dt * W_v
        v_path[t+1] = max(v_t + dv, 0)

        d_log_S = (r - q - 0.5 * v_t) * dt + sqrt_v * sqrt_dt * W_S
        S_path[t+1] = S_t * np.exp(d_log_S) 
    
    return S_path, v_path

params = {
        'S0': 100,
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'xi': 0.3,
        'cor': -0.7,
        'r': 0.02,
        'q': 0.01
    }
    
S, v = simulate_heston_path(params, T=1.0, N=252)
