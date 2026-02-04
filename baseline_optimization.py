from scipy.optimize import minimize
from loss_functions import squad_error_for_all_options
import numpy as np
import pandas as pd

def calibrate_least_squares(market_data, initial_guess=None, method='L-BFGS-B'):
    '''
    калибровка модели Хекстона методом наименьших квадратов
    '''

    # начальные значения параметров {v0, kappa, theta, xi, cor}
    if initial_guess is None:
        x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.7])
    else:
        x0 = initial_guess

    bounds = [(0.001, 1.0), (0.1, 10.0), (0.001, 0.5), (0.01, 1.0), (-0.99, 0.99)]
    
    # в loss_func_model константны начальная цена акции(S0), процентная ставка r, девидендная доходность q
    def loss_func_model(params_optimize):
        params_stock = {
            'v0': params_optimize[0],
            'kappa': params_optimize[1],
            'theta': params_optimize[2],
            'xi': params_optimize[3],
            'cor': params_optimize[4]
        }

        return squad_error_for_all_options(params_stock, market_data)
    
    result = minimize(
        loss_func_model,
        x0,
        method=method,
        bounds=bounds,
        options={'maxiter': 50, 'disp': True}
    )
    
    return result

market_data = pd.read_csv("dataset/dataset_with_noisy.csv")
x0 = np.array([0.02, 1.0, 0.14, 0.5, 0.3])
print(calibrate_least_squares(market_data, x0))