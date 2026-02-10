import numpy as np
from sup import AgentDQN, HedgingEnv

def main():
    print("🚀 ЗАПУСК ПЕРВОГО ОБУЧЕНИЯ RL ДЛЯ ХЕДЖИРОВАНИЯ")
    print("="*50)
    
    # 1. СОЗДАЕМ СРЕДУ
    print("\n1. Создаем среду хеджирования...")
    params = [0.04, 2.0, 0.04, 0.3, -0.7]
    env = HedgingEnv(
        S0=150.0,
        T=30/365,  # 30 дней
        K=155.0,
        q=0.0,
        r=0.02,
        params_option=params
    )
    print(f"   ✓ Среда создана")
    print(f"   ✓ Размер state: {env.observation_space.shape[0]}")
    print(f"   ✓ Доступно действий: {env.action_space.n}")
    
    # 2. СОЗДАЕМ АГЕНТА
    print("\n2. Создаем DQN агента...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = AgentDQN(state_dim=state_dim, action_dim=action_dim)
    print(f"   ✓ Агент создан")
    print(f"   ✓ Начальный epsilon: {agent.epsilon}")
    print(f"   ✓ Batch size: {agent.batch_size}")
    
    # 3. ПРОВЕРЯЕМ СРЕДУ (1 эпизод со случайными действиями)
    print("\n3. Тестируем среду (1 эпизод со случайными действиями)...")
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        # Случайное действие
        action = np.random.randint(0, 3)
        
        # Шаг в среде
        next_state, reward, done = env.step(action)
        
        # Сохраняем опыт (для обучения)
        agent.remember(state, action, reward, next_state, done)
        
        # Обновляем
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"   ✓ Эпизод завершен за {steps} шагов")
    print(f"   ✓ Итоговая награда: {total_reward:.2f}")
    print(f"   ✓ В памяти: {len(agent.memory)} примеров")
    
    # 4. ПЕРВОЕ ОБУЧЕНИЕ (если накопили достаточно примеров)
    print("\n4. Пробуем обучить агента...")
    if len(agent.memory) >= agent.batch_size:
        loss = agent.learn_from_memory()
        print(f"   ✓ Первое обучение прошло!")
        print(f"   ✓ Loss: {loss:.6f}")
        print(f"   ✓ Новый epsilon: {agent.epsilon:.4f}")
    else:
        print(f"   ⚠️  Недостаточно данных для обучения")
        print(f"   ⚠️  Нужно: {agent.batch_size}, есть: {len(agent.memory)}")
    
    # 5. ОБУЧАЕМ НА НЕСКОЛЬКИХ ЭПИЗОДАХ
    print("\n5. Обучаем на 10 эпизодах...")
    print("-"*40)
    
    for episode in range(1, 11):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Агент выбирает действие (с exploration!)
            action = agent.act(state)
            
            # Шаг в среде
            next_state, reward, done = env.step(action)
            
            # Сохраняем опыт
            agent.remember(state, action, reward, next_state, done)
            
            # Периодически обучаем
            if episode_steps % 4 == 0:  # Обучаем каждый 4-й шаг
                agent.learn_from_memory()
            
            # Обновляем
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # Выводим прогресс
        print(f"Эпизод {episode:2d}: Награда = {episode_reward:7.2f}, "
              f"Шагов = {episode_steps:3d}, "
              f"Epsilon = {agent.epsilon:.3f}, "
              f"Память = {len(agent.memory)}")
    
    print("-"*40)
    
    # 6. ТЕСТИРУЕМ ОБУЧЕННОГО АГЕНТА
    print("\n6. Тестируем обученного агента (без exploration)...")
    test_rewards = []
    
    for test_ep in range(3):
        state = env.reset()
        test_reward = 0
        done = False
        
        # Временно отключаем exploration
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        while not done:
            action = agent.act(state)  # Теперь только exploitation
            state, reward, done = env.step(action)
            test_reward += reward
        
        # Восстанавливаем epsilon
        agent.epsilon = old_epsilon
        
        test_rewards.append(test_reward)
        print(f"   Тест {test_ep+1}: Награда = {test_reward:.2f}")
    
    # 7. СОХРАНЯЕМ МОДЕЛЬ
    print("\n7. Сохраняем модель...")
    agent.save("my_first_dqn_agent.pth")
    print("   ✓ Модель сохранена в 'my_first_dqn_agent.pth'")
    
    print("\n" + "="*50)
    print("🎉 ВАШ ПЕРВЫЙ RL АГЕНТ ОБУЧЕН!")
    print("="*50)
    print("\nЧто дальше:")
    print("1. Запустите этот скрипт: python first_training.py")
    print("2. Посмотрите на награды - они должны улучшаться")
    print("3. Попробуйте увеличить число эпизодов до 50-100")
    print("4. Экспериментируйте с гиперпараметрами!")

if __name__ == "__main__":
    main()