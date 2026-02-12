# fine_tune_101.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from RL_AGENT_AND_ENVIRONMENT import HedgingEnv, AgentDQN
from simulate_hekston import compute_price_call_single
import os
from collections import deque

def fine_tune_agent():
    print("="*70)
    print("🚀 FINE-TUNING: ДООБУЧАЕМ АГЕНТА НА СЛУЧАЙНЫХ ТРАЕКТОРИЯХ")
    print("="*70)
    
    # ============= 1. ПАРАМЕТРЫ =============
    params = [0.04, 2.0, 0.04, 0.3, -0.7]
    S0 = 150.0
    K = 155.0
    T = 30/365
    r = 0.02
    q = 0.0
    
    # ============= 2. СОЗДАЕМ СРЕДУ =============
    env = HedgingEnv(
        S0=S0, T=T, K=K, q=q, r=r, 
        params_option=params
    )
    
    # ============= 3. ЗАГРУЖАЕМ ОБУЧЕННУЮ МОДЕЛЬ =============
    print("\n📂 Загрузка обученной модели...")
    agent = AgentDQN(state_dim=6, action_dim=101)
    
    try:
        agent.load("best_agent_101.pth")
        print(f"✅ Модель загружена!")
        print(f"   Текущий epsilon: {agent.epsilon}")
        print(f"   Размер памяти: {len(agent.memory)}")
    except:
        print("❌ Ошибка: файл best_agent_101.pth не найден!")
        return
    
    # ============= 4. НАСТРОЙКИ ДЛЯ FINE-TUNING =============
    # УВЕЛИЧИВАЕМ EPSILON ДЛЯ ИССЛЕДОВАНИЯ!
    agent.epsilon = 0.3          # Начинаем с 30% случайных действий
    agent.epsilon_min = 0.01    # Минимум 1%
    agent.epsilon_decay = 0.997 # Медленный спад
    
    # УМЕНЬШАЕМ LEARNING RATE ДЛЯ ТОНКОЙ НАСТРОЙКИ
    agent.learning_rate = 0.0005  # В 2 раза меньше!
    agent.optimizer = torch.optim.Adam(
        agent.policy_net.parameters(), 
        lr=agent.learning_rate
    )
    
    # УВЕЛИЧИВАЕМ ПАМЯТЬ
    agent.memory = deque(maxlen=10000)
    
    print(f"\n⚙️  Настройки fine-tuning:")
    print(f"   Epsilon: {agent.epsilon} → {agent.epsilon_min}")
    print(f"   Learning rate: {agent.learning_rate}")
    print(f"   Batch size: {agent.batch_size}")
    print(f"   Memory size: {agent.memory.maxlen}")
    
    # ============= 5. ДООБУЧЕНИЕ =============
    print("\n📚 ЭТАП: ДООБУЧЕНИЕ НА СЛУЧАЙНЫХ ТРАЕКТОРИЯХ")
    print("-"*70)
    
    episodes = 500  # Достаточно для fine-tuning!
    best_reward = -np.inf
    rewards_history = []
    hedge_errors_history = []
    
    for episode in range(1, episodes + 1):
        # КАЖДЫЙ ЭПИЗОД - НОВЫЙ СЛУЧАЙНЫЙ SEED!
        # env.reset() уже использует случайный seed
        
        state = env.reset()
        total_reward = 0
        episode_hedge_errors = []
        done = False
        
        while not done:
            # Агент выбирает действие (с exploration!)
            action = agent.act(state)
            
            # Шаг в среде
            next_state, reward, done = env.step(action)
            
            # Сохраняем опыт
            agent.remember(state, action, reward, next_state, done)
            
            # Обучаемся
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn_from_memory()
            
            # Собираем метрики
            total_reward += reward
            episode_hedge_errors.append(abs(env.hedge_error))
            state = next_state
        
        # Сохраняем историю
        rewards_history.append(total_reward)
        avg_hedge_error = np.mean(episode_hedge_errors) if episode_hedge_errors else 0
        hedge_errors_history.append(avg_hedge_error)
        
        # Сохраняем лучшую модель
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("best_agent_101_finetuned.pth")
        
        # Прогресс каждые 20 эпизодов
        if episode % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
            avg_hedge = np.mean(hedge_errors_history[-20:]) if len(hedge_errors_history) >= 20 else np.mean(hedge_errors_history)
            
            print(f"   Эпизод {episode:3d}/500 | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Hedge Error: {avg_hedge:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory)}")
    
    # ============= 6. СОХРАНЯЕМ ФИНАЛЬНУЮ МОДЕЛЬ =============
    agent.save("agent_101_finetuned_final.pth")
    print(f"\n✅ Fine-tuning завершен!")
    print(f"   Лучшая награда: {best_reward:.2f}")
    print(f"   Финальный epsilon: {agent.epsilon:.4f}")
    
    # ============= 7. ТЕСТИРОВАНИЕ =============
    print("\n🧪 ЭТАП: ТЕСТИРОВАНИЕ ДООБУЧЕННОГО АГЕНТА")
    print("-"*70)
    
    # Отключаем exploration для теста
    agent.epsilon = 0.0
    
    test_rewards = []
    test_hedge_errors = []
    
    for test in range(50):
        state = env.reset()
        total_reward = 0
        hedge_errors = []
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            hedge_errors.append(abs(env.hedge_error))
        
        test_rewards.append(total_reward)
        test_hedge_errors.append(np.mean(hedge_errors))
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   Средняя награда: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"   Средняя ошибка хеджа: {np.mean(test_hedge_errors):.4f}")
    print(f"   Лучшая награда: {max(test_rewards):.2f}")
    print(f"   Худшая награда: {min(test_rewards):.2f}")
    
    # ============= 8. ВИЗУАЛИЗАЦИЯ ПРОГРЕССА =============
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, alpha=0.6, label='Reward per episode')
    
    # Скользящее среднее
    if len(rewards_history) >= 20:
        moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(rewards_history)), moving_avg, 'r-', linewidth=2, label='Moving avg (20)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Fine-tuning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(hedge_errors_history, alpha=0.6, label='Hedge Error')
    
    if len(hedge_errors_history) >= 20:
        moving_hedge = np.convolve(hedge_errors_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(hedge_errors_history)), moving_hedge, 'r-', linewidth=2, label='Moving avg (20)')
    
    plt.xlabel('Episode')
    plt.ylabel('Hedge Error')
    plt.title('Hedge Error During Fine-tuning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fine_tuning_progress.png', dpi=150)
    plt.show()
    
    print(f"\n📊 Графики сохранены в 'fine_tuning_progress.png'")
    print("="*70)
    print("✅ FINE-TUNING ЗАВЕРШЕН УСПЕШНО!")
    print("="*70)
    
    return agent

def quick_compare():
    """Быстрое сравнение до и после fine-tuning"""
    print("\n🔍 БЫСТРОЕ СРАВНЕНИЕ:")
    print("-"*70)
    
    params = [0.04, 2.0, 0.04, 0.3, -0.7]
    env = HedgingEnv(S0=150.0, T=30/365, K=155.0, q=0.0, r=0.02, params_option=params)
    
    # Загружаем ОБЕ модели
    agent_old = AgentDQN(state_dim=6, action_dim=101)
    agent_new = AgentDQN(state_dim=6, action_dim=101)
    
    try:
        agent_old.load("best_agent_101.pth")
        agent_new.load("best_agent_101_finetuned.pth")
    except:
        print("❌ Не удалось загрузить модели для сравнения")
        return
    
    # Отключаем exploration
    agent_old.epsilon = 0.0
    agent_new.epsilon = 0.0
    
    # Тестируем на 10 случайных траекториях
    old_rewards = []
    new_rewards = []
    
    for _ in range(10):
        # Одинаковый seed для честного сравнения
        seed = np.random.randint(0, 10000)
        
        # Тест старой модели
        np.random.seed(seed)
        state = env.reset()
        old_reward = 0
        done = False
        while not done:
            action = agent_old.act(state)
            state, reward, done = env.step(action)
            old_reward += reward
        old_rewards.append(old_reward)
        
        # Тест новой модели
        np.random.seed(seed)
        state = env.reset()
        new_reward = 0
        done = False
        while not done:
            action = agent_new.act(state)
            state, reward, done = env.step(action)
            new_reward += reward
        new_rewards.append(new_reward)
    
    print(f"\n📊 СРАВНЕНИЕ НА ОДИНАКОВЫХ ТРАЕКТОРИЯХ:")
    print(f"   До fine-tuning:  {np.mean(old_rewards):.2f} ± {np.std(old_rewards):.2f}")
    print(f"   После fine-tuning: {np.mean(new_rewards):.2f} ± {np.std(new_rewards):.2f}")
    print(f"   Улучшение: {((np.mean(new_rewards) - np.mean(old_rewards)) / abs(np.mean(old_rewards)) * 100):+.1f}%")

if __name__ == "__main__":
    # ЗАПУСК FINE-TUNING
    agent = fine_tune_agent()
    
    # БЫСТРОЕ СРАВНЕНИЕ
    quick_compare()