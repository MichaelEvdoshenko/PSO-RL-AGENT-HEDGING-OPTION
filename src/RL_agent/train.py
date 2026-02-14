import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RL_agent.RL_AGENT_AND_ENVIRONMENT import HedgingEnv, AgentDQN
from RL_agent.hekston_model import compute_price_call_single, calculate_heston_delta
from collections import deque

def train(params, S0, K, T, r, q):
    #params = [0.04, 2.0, 0.04, 0.3, -0.7]
    #S0 = 150.0
    #K = 135.0
    #T = 90/365
    #r = 0.02
    #q = 0.03
    
    env = HedgingEnv(S0=S0, T=T, K=K, q=q, r=r, params_option=params)
    agent = AgentDQN(state_dim=6, action_dim=101)
    
    agent.epsilon = 0.7
    agent.epsilon_decay = 0.999
    agent.learning_rate = 0.001
    agent.batch_size = 300
    agent.memory = deque(maxlen=10000)
    
    episodes = 2000
    rewards_history = []
    best_reward = -np.inf
    sup_count = 0

    for episode in range(1, episodes + 1):
        sup_count+=1
        state = env.reset()
        total_reward = 0
        done = False
        agent.decay_epsilon()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.learn_from_memory()
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f"best_agent_101_{episode}.pth")
            sup_count = 0

        if sup_count == 150:
            agent.save(f"checkpoint_{episode}.pth")
            sup_count = 0
        
        if episode % 2 == 0:
            avg_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
            print(f"   Эпизод {episode:4d}: Reward = {total_reward:8.2f} | "
                  f"Avg = {avg_reward:8.2f} | Epsilon = {agent.epsilon:.3f} | "
                  f"Memory = {len(agent.memory)}" f"sup_count = {sup_count}")
