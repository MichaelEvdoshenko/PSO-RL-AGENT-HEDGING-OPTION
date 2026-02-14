from PSO.PSO import PSO_optimize
from RL_agent.RL_AGENT_AND_ENVIRONMENT import HedgingEnv, AgentDQN
from RL_agent.test import test
from RL_agent.train import train
import numpy as np
import pandas as pd

if __name__ == "__main__":
    market_data = pd.read_csv("../dataset/dataset.csv")
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
    "АЛГОРИТМ ВЫПОЛНЯЕТСЯ ЗА 1-2 ЧАСА"
    #best_params, best_error = a.PSO_algorithm() 
    #print(f"\nЛучшие параметры: {best_params}")
    #print(f"Лучшая ошибка: {best_error}")

    best_params = [0.04, 2.0, 0.04, 0.3, -0.7]

    "АГЕНТ ОБУЧАЕТСЯ ЗА 15-20 ЧАСОВ"
    #train(best_params, S0=150, K=135, T=90/365, r=0.02, q=0.03)
        #S0 - начальная цена рискового актива
        #K - страйк опциона
        #T - время до экспирации в годах
        #r - процентная ставка
        #q - дивидендная доходность

    agent = AgentDQN()
    agent.load("best_agent.pth") #уже предобученный мной агент
    test(best_params, 150.0, 135.0, 90/365, 0.02, 0.03, agent) #получаем оптимальную стратегию хеджирования для случайного пути рискового актива
