import numpy as np
import pandas as pd
from loss_functions import heston_loss_function_stable
from add_param import add_param

class PSO_optimize:
    def __init__(self,
                 bounds,
                 market_data,
                 n_particles=50,
                 max_iter=100,
                 w_start=0.9,
                 w_end=0.4,
                 c1_start=2.0,
                 c1_end=0.5,
                 c2_start=2.0,
                 c2_end=0.5):
        
        self.count_param = len(bounds)
        self.market_data = market_data
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        
        self.w_start = w_start
        self.w_end = w_end
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end

    def PSO_algorithm(self):
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        ranges = highs - lows

        all_x0 = np.random.uniform(
            low=lows,
            high=highs,
            size=(self.n_particles, self.count_param)
        )

        all_v0 = np.random.uniform(
            low=-ranges,
            high=ranges,
            size=(self.n_particles, self.count_param)
        )

        market_data_prepared = add_param(self.market_data)

        current_error_all_x = np.zeros(self.n_particles)
        for i, x0 in enumerate(all_x0):
            current_error_all_x[i] = heston_loss_function_stable(x0, market_data_prepared)
        
        best_error_all_x = current_error_all_x.copy()
        best_x_all = all_x0.copy()
        
        global_best_idx = np.argmin(best_error_all_x)
        global_best_x = best_x_all[global_best_idx].copy()
        global_best_error = best_error_all_x[global_best_idx]
        
        stagnation_counters = np.zeros(self.n_particles)
        previous_global_best_error = global_best_error
        global_stagnation_counter = 0
        
        for step in range(self.max_iter):
            progress = step / self.max_iter
            current_w = self.w_start - (self.w_start - self.w_end) * progress**2
            current_c1 = self.c1_start + (self.c1_end - self.c1_start) * progress
            current_c2 = self.c2_start + (self.c2_end - self.c2_start) * progress

            r1 = np.random.uniform(0, 1, size=(self.n_particles, self.count_param))
            r2 = np.random.uniform(0, 1, size=(self.n_particles, self.count_param))
            
            all_v0 = (current_w * all_v0 + current_c1 * r1 * (best_x_all - all_x0) + current_c2 * r2 * (np.tile(global_best_x, (self.n_particles, 1)) - all_x0))
            
            max_velocity = 0.2 * ranges
            all_v0 = np.clip(all_v0, -max_velocity, max_velocity)
            
            all_x0 += all_v0
            
            for d in range(self.count_param):
                mask_low = all_x0[:, d] < lows[d]
                mask_high = all_x0[:, d] > highs[d]
                
                all_x0[mask_low, d] = 2 * lows[d] - all_x0[mask_low, d]
                all_x0[mask_high, d] = 2 * highs[d] - all_x0[mask_high, d]
                
                all_x0[:, d] = np.clip(all_x0[:, d], lows[d], highs[d])
            
            for i, x0 in enumerate(all_x0):
                current_error_all_x[i] = heston_loss_function_stable(x0, market_data_prepared)
                
                if current_error_all_x[i] < best_error_all_x[i]:
                    best_x_all[i] = x0.copy()
                    best_error_all_x[i] = current_error_all_x[i]
                    stagnation_counters[i] = 0
                else:
                    stagnation_counters[i] += 1
                
                if current_error_all_x[i] < global_best_error:
                    global_best_x = x0.copy()
                    global_best_error = current_error_all_x[i]
                    global_stagnation_counter = 0
                    global_best_idx = i
            
            if np.abs(global_best_error - previous_global_best_error) < 3:
                global_stagnation_counter += 1
            else:
                global_stagnation_counter = 0
                previous_global_best_error = global_best_error
            
            n_restarted = 0
            for i in range(self.n_particles):
                if i == global_best_idx:
                    if stagnation_counters[i] > 10:
                        avg_error = 0
                        for _ in range(30):
                            avg_error += heston_loss_function_stable(all_x0[i], market_data_prepared)
                        avg_error /= 30
            
                        if avg_error / global_best_error > 4.0:
                            print(f"Iter {step}: Global best particle seems lucky! {global_best_error:.2f}, Avg error: {avg_error:.2f}")
                
                            noise_scale = 0.3 * ranges * (1.0 - progress)

                            new_position = np.random.uniform(lows, highs)
                
                            new_position = np.clip(new_position, lows, highs)

                            all_x0[i] = new_position
                            all_v0[i] = np.random.uniform(-ranges, ranges) * 0.5

                            new_error = heston_loss_function_stable(new_position, market_data_prepared)
                            current_error_all_x[i] = new_error

                            best_x_all[i] = new_position.copy()
                            best_error_all_x[i] = new_error

                            for sss, x0 in enumerate(all_x0):
                                if current_error_all_x[sss] < global_best_error:
                                    global_best_x = x0.copy()
                                    global_best_error = current_error_all_x[sss]
                                    global_best_idx = sss

                            stagnation_counters[i] = 0

                            best_error_all_x[i] = min(best_error_all_x[i], new_error)
                
                            print(f"  Reinitialized global best particle. New error: {new_error:.2f}")
                        else:
                            stagnation_counters[i] = 0

                if stagnation_counters[i] > 10:
                    if np.random.random() < 0.7:

                        direction = global_best_x - all_x0[i]
                        noise_scale = 0.1 * ranges * (1.0 - progress)
                        random_component = np.random.normal(0, noise_scale)
                        
                        new_position = all_x0[i] + 0.5 * direction + 0.5 * random_component
                    else:
                        noise_scale = 0.2 * ranges * (1.0 - progress)
                        new_position = all_x0[i] + np.random.normal(0, noise_scale)
                    
                    new_position = np.clip(new_position, lows, highs)
                    
                    all_x0[i] = new_position
                    all_v0[i] = np.random.uniform(-ranges, ranges) * 0.5
                    
                    current_error = heston_loss_function_stable(new_position, market_data_prepared)
                    current_error_all_x[i] = current_error
                    
                    if current_error < best_error_all_x[i]:
                        best_x_all[i] = new_position.copy()
                        best_error_all_x[i] = current_error
                        global_best_idx = i 
                    
                    stagnation_counters[i] = 0
                    n_restarted += 1
            
            print(f"{step:3d}: Best error = {global_best_error:.2f}, "f"w={current_w:.2f}, c1={current_c1:.2f}, c2={current_c2:.2f}, "f"Restarted: {n_restarted}, {global_best_x}")
        
        print(f"\nФинальная лучшая ошибка: {global_best_error:.2f}")
        print(f"Финальные параметры: {global_best_x}")
        
        return global_best_x, global_best_error

market_data = pd.read_csv("dataset/dataset_with_noisy.csv")
bounds = [
    [0.01, 0.7],
    [0.5, 5.0],
    [0.01, 0.7],
    [0.1, 2.0],
    [-0.9, -0.1]
]

a = PSO_optimize(
    bounds=bounds,
    market_data=market_data,
    n_particles=30,
    max_iter=100,
    w_start=0.9,
    w_end=0.4,
    c1_start=2.0,
    c1_end=0.5,
    c2_start=2.0,
    c2_end=0.5
)

#best_params, best_error = a.PSO_algorithm()
#print(f"\nЛучшие параметры: {best_params}")
#print(f"Лучшая ошибка: {best_error}")

params = np.array([ 0.0420915, 2.33998196, 0.03772801, 0.27454914, -0.74216899])
market_data = pd.read_csv("dataset/dataset_with_noisy.csv")
added = add_param(market_data)

print(heston_loss_function_stable(params, added))
print(heston_loss_function_stable(np.array([0.04,2.0,0.04,0.3,-0.7]), added))