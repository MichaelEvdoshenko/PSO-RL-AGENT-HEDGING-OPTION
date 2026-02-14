import pandas as pd
import numpy as np
from simulate_hekston import compute_price_call
from BSM import compute_vol_BSM

def heston_loss_function_stable(S0, params_stock, market_data, seed=42, n_mc_samples=10):

    v0, kappa, theta, sigma_v, cor = params_stock

    penalty = 0.0

    if v0 < 0.005 or v0 > 0.5:
        penalty += 100000.0 * ((v0 - 0.1)**2 + 1)
    
    if theta < 0.005 or theta > 0.5:
        penalty += 100000.0 * ((theta - 0.1)**2 + 1)
    
    if kappa < 0.5 or kappa > 10.0:
        penalty += 100000.0 * ((kappa - 3.0)**2 + 1)
    
    if sigma_v < 0.1 or sigma_v > 2.0:
        penalty += 100000.0 * ((sigma_v - 0.3)**2 + 1)
    
    if cor > -0.1 or cor < -0.9:
        penalty += 100000.0 * ((cor + 0.7)**2 + 1)

    feller_condition = sigma_v**2 - 2*kappa*theta
    if feller_condition > 0:
        penalty += 500000.0 * feller_condition

    if penalty > 100000:
        return penalty
    
    all_iv_errors = []
    
    for mc_run in range(n_mc_samples):
        mc_seed = seed + mc_run * 1000
        #np.random.seed(mc_seed)
        
        heston_prices = compute_price_call(S0,
            {'v0': v0, 'kappa': kappa, 'theta': theta, 
             'xi': sigma_v, 'cor': cor},
            market_data
        )
        
        iv_errors_run = []
        for price, (_, row) in zip(heston_prices, market_data.iterrows()):
            single_params = {
                "S0": row['S0'], "K": row['K'], 
                "T": row['T'], "r": row['r'], "q": row['q']
            }
            
            iv = compute_vol_BSM(single_params, float(price))
            if not np.isnan(iv):
                market_iv = row['market_iv']
                if market_iv > 0.01:
                    rel_error = abs(iv - market_iv) / market_iv
                    moneyness = row['K'] / row['S0']
                    weight = np.exp(-5 * (moneyness - 1)**2)
                    iv_errors_run.append(weight * rel_error)
        
        if iv_errors_run:
            all_iv_errors.append(np.median(iv_errors_run))
    
    if all_iv_errors:
        iv_error = np.mean(all_iv_errors)
    else:
        iv_error = 1.0
    
    if len(all_iv_errors) > 1:
        mc_std = np.std(all_iv_errors)
        stability_penalty = 10000.0 * mc_std
        penalty += stability_penalty
    
    if kappa < 1.0:
        penalty += 50000.0 * (1.0 - kappa)

    if cor > -0.3:
        penalty += 30000.0 * (cor + 0.3)**2
    
    total_loss = 0.1 * iv_error + 0.9 * penalty / 100000.0
    
    return total_loss