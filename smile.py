import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSM import compute_vol_BSM

dataset = pd.read_csv("dataset/dataset_with_noisy.csv")
IV = []
for index, row in dataset.iterrows():
    params_dict = row.to_dict()
    IV.append(compute_vol_BSM(params_dict, dataset["call_mid"].iloc[index]))
dataset['IV'] = np.array(IV)

print(dataset[["S0","K","T","r","q", "call_mid", "IV"]].head(10))

dataset_T_1 = dataset.loc[((dataset["T"] == 1.0) & (dataset["IV"].notna()))]
coeffs = np.polyfit(dataset_T_1["IV"], dataset_T_1["K"], 4)
poly = np.poly1d(coeffs)


x_fit = np.linspace(np.min(dataset_T_1["IV"]), np.max(dataset_T_1["IV"]), 100)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit)

dataset_T_1 = dataset.loc[((dataset["T"] == 1.5) & (dataset["IV"].notna()))]
coeffs = np.polyfit(dataset_T_1["IV"], dataset_T_1["K"], 4)
poly = np.poly1d(coeffs)


x_fit = np.linspace(np.min(dataset_T_1["IV"]), np.max(dataset_T_1["IV"]), 100)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit)

dataset_T_1 = dataset.loc[((dataset["T"] == 2.0) & (dataset["IV"].notna()))]
coeffs = np.polyfit(dataset_T_1["IV"], dataset_T_1["K"], 4)
poly = np.poly1d(coeffs)


x_fit = np.linspace(np.min(dataset_T_1["IV"]), np.max(dataset_T_1["IV"]), 100)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit)
plt.show()

