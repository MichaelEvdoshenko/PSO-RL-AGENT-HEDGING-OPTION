import numpy as np
import pandas as pd
from loss_functions import squad_error_for_all_options

class PSO_optimize:
    def __init__(self,
             bounds, #массив [[min1, max1], [min2, max2],...]
             market_data, #dataset_with_noisy
             n_particles=50, #количество частиц
             max_iter=100, #максимальное количество шагов алгоритма
             w=0.9, #инерция в формуле PSO
             c1=0.5, #коэффициент обучения в формуле PSO 
             c2=0.3 #коэффициент обучения в формуле PSO
            ):
        self.count_param = len(bounds)
        self.market_data = market_data
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w 
        self.c1 = c1
        self.c2 = c2

    def PSO_algorithm(self):
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]

        all_x0 = np.random.uniform(
            low=lows,
            high=highs,
            size=(self.n_particles, len(self.bounds))
        )

        ranges = highs - lows
        all_v0 = np.random.uniform(
            low=-ranges,
            high=ranges,
            size=(self.n_particles, len(self.bounds))
        )


        current_error_all_x = np.zeros(self.n_particles)
        
        all_x0 += all_v0
        for d in range(self.count_param):
                all_x0[:, d] = np.clip(all_x0[:, d], lows[d], highs[d])
    
        for i, x0 in enumerate(all_x0):
            current_error_all_x[i] = squad_error_for_all_options(x0, market_data)
        
        best_error_all_x = current_error_all_x.copy()
        best_x = all_x0[np.argmin(current_error_all_x)]
        best_x_all = all_x0.copy()

        for step in range(self.max_iter):
            r1 = np.random.uniform(0, 1, size=(self.n_particles, self.count_param))
            r2 = np.random.uniform(0, 1, size=(self.n_particles, self.count_param))
            if step > 200 and step % 50 == 0:
                worst_indices = np.argsort(best_error_all_x)[-n_particles//3:]
                for idx in worst_indices:
                    all_x0[idx] = np.random.uniform(lows, highs)
            current_w = 0.9 * (1 - (step/self.max_iter)**2)

            all_v0 = current_w * all_v0 + self.c1 * r1 * (best_x_all - all_x0) + self.c2 * r2 * (np.tile(best_x, (self.n_particles, 1)) - all_x0)
            if np.random.random() < 0.1:
                all_v0 += np.random.normal(0, 0.01, size=all_v0.shape)    
            all_x0 += all_v0

            for d in range(self.count_param):
                all_x0[:, d] = np.clip(all_x0[:, d], lows[d], highs[d])
    
            for i, x0 in enumerate(all_x0):
                current_error_all_x[i] = squad_error_for_all_options(x0, market_data)
                if (current_error_all_x[i] < best_error_all_x[i]):
                    best_x_all[i] = x0
                    best_error_all_x[i] = current_error_all_x[i]
            
            best_x = best_x_all[np.argmin(best_error_all_x)]
            print(step)
        return best_x, self.market_data.iloc[0]

market_data = pd.read_csv("dataset/dataset_with_noisy.csv")
bounds = [
    [0.01, 0.9],
    [0.5, 10.0],
    [0.01, 0.9],
    [0.1, 2.0],
    [-0.99, 0.0]
]
a = PSO_optimize(bounds, market_data, 50)
print(a.PSO_algorithm())