import numpy as np
import pandas as pd
from BSM import compute_vol_BSM

def add_param(market_data):
    """
    Добавляет вычисленные параметры к рыночным данным
    берём только ATM опционы
    
    market_data: DataFrame с колонками:
        S0, K, T, r, q, call_bid, call_ask, call_mid
    """
    
    df = market_data.copy()
    
    df['bid_ask_spread'] = df['call_ask'] - df['call_bid']
    df['weight'] = 1.0 / (df['bid_ask_spread'] + 1e-6)

    IV_list = []
    
    for index, row in df.iterrows():
        params_dict = {
            "S0": row['S0'],
            "K": row['K'],
            "T": row['T'],
            "r": row['r'],
            "q": row['q']
        }
        
        market_price = row['call_mid']
        
        iv = compute_vol_BSM(params_dict, market_price)
        IV_list.append(iv)
    
    df['market_iv'] = IV_list

    df_clean = df[df['market_iv'].notna()].copy()

    atm_mask = (df['K'] / df['S0']).between(0.9, 1.1)
    df_clean = df_clean.loc[atm_mask]
    return df_clean