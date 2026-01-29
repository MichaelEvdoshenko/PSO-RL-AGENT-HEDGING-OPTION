from func_generate import compute_price_call 
import pandas as pd
import numpy as np

def generate_dataset(true_params, expires, strikes, N=250, n_paths=50000):
    data = []
    for E in expires:
        for K in strikes:
            print(E, K)
            params_option = {'T': E, 'K': K}
            S_0, v = compute_price_call(true_params, params_option, N=N, n_paths=n_paths)
            data.append({
                'call_price_clean': S_0,
                'T': E,
                'K': K,
                'true_params': str(true_params)
            })
    df = pd.DataFrame(data)
    df_noisy = df.copy()
    
    noise = np.random.normal(0, 0.01, len(df))
    df_noisy['call_price_market'] = df_noisy['call_price_clean'] * (1 + noise)
    
    spread = 0.03
    df_noisy['call_bid'] = df_noisy['call_price_market'] * (1 - spread/2)
    df_noisy['call_ask'] = df_noisy['call_price_market'] * (1 + spread/2)
    df_noisy['call_mid'] = (df_noisy['call_bid'] + df_noisy['call_ask']) / 2
    
    return df_noisy, df

TRUE_HESTON_PARAMS = {
    'S0': 100,
    'v0': 0.04,
    'kappa': 2.0,
    'theta': 0.04,
    'xi': 0.3,
    'cor': -0.7,
    'r': 0.02,
    'q': 0.01
}

expires = np.array([1/12, 2/12, 3/12, 4/12, 6/12, 9/12, 1.0, 1.5, 2.0, 3.0])
strikes = TRUE_HESTON_PARAMS['S0'] * np.array([0.70, 0.75, 0.80, 0.81, 0.82, 0.84, 0.86, 0.88, 0.89, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.30])

df_noisy, df = generate_dataset(TRUE_HESTON_PARAMS, expires, strikes)
df_noisy.to_csv('dataset_with_noisy.csv', index=False)