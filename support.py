import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from loss_functions import heston_loss_function
from add_param import add_param

market_data = pd.read_csv("dataset/dataset_with_noisy.csv")
market_data_added = add_param(market_data)

errors = []
params = np.array([0.04, 2.0, 0.04, 0.3, -0.7])
lists = np.linspace(0.0, 0.3, 100)
for i in lists:
    params[0] = i
    errors.append(heston_loss_function(params, market_data_added))
    print(i)

plt.plot(lists, errors)
plt.show()