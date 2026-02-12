import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RL_AGENT_AND_ENVIRONMENT import HedgingEnv, AgentDQN
from simulate_hekston import compute_price_call_single, calculate_heston_delta
from collections import deque

def train_and_visualize():
    print("🚀 ОБУЧЕНИЕ АГЕНТА И ВИЗУАЛИЗАЦИЯ ХЕДЖИРОВАНИЯ (101 ДЕЙСТВИЕ)")
    print("="*70)
    
    # ============= 1. ПАРАМЕТРЫ =============
    params = [0.04, 2.0, 0.04, 0.3, -0.7]
    S0 = 150.0
    K = 155.0
    T = 30/365
    r = 0.02
    q = 0.0
    
    # ============= 2. ОБУЧАЕМ АГЕНТА =============
    print("\n📚 ЭТАП 1: ОБУЧЕНИЕ АГЕНТА")
    print("-"*50)
    
    env = HedgingEnv(S0=S0, T=T, K=K, q=q, r=r, params_option=params)
    agent = AgentDQN(state_dim=6, action_dim=101)  # ← 101 ДЕЙСТВИЕ!
    
    # Настройки для 101 действия
    agent.epsilon = 0.5          # Больше exploration
    agent.epsilon_decay = 0.997  # Медленнее decay
    agent.learning_rate = 0.001  # Меньше learning rate
    agent.batch_size = 100        # Больше batch
    agent.memory = deque(maxlen=10000)  # Больше памяти
    
    episodes = 1000  # Минимум 500 эпизодов
    rewards_history = []
    best_reward = -np.inf
    
    for episode in range(1, episodes + 1):
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
        
        # Сохраняем лучшую модель
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("best_agent_101.pth")
        
        if episode % 2 == 0:
            avg_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
            print(f"   Эпизод {episode:4d}: Reward = {total_reward:8.2f} | "
                  f"Avg = {avg_reward:8.2f} | Epsilon = {agent.epsilon:.3f} | "
                  f"Memory = {len(agent.memory)}")
    
    agent.epsilon = 0.0
    agent.load("best_agent_101.pth")  # Загружаем лучшую модель!
    print(f"\n✅ Обучение завершено! Лучшая награда: {best_reward:.2f}")
    
    # ============= 3. СИМУЛЯЦИЯ =============
    print("\n📈 ЭТАП 2: СИМУЛЯЦИЯ ХЕДЖИРОВАНИЯ")
    print("-"*50)
    
    sim_env = HedgingEnv(S0=S0, T=T, K=K, q=q, r=r, params_option=params)
    state = sim_env.reset()
    
    # Массивы для данных
    days = []
    stock_prices = []
    option_prices = []
    deltas = []
    hedge_positions = []
    actions_history = []
    actions_percent = []  # ← Для 101 действия!
    portfolio_values = []
    hedge_errors = []
    
    day = 0
    done = False
    
    print("   Симуляция 30 дней хеджирования с 101 возможным действием...")
    
    while not done:
        days.append(day)
        stock_prices.append(sim_env.current_price_stock)
        
        option_price = compute_price_call_single(
            sim_env.current_price_stock, K, 
            max(sim_env.rest_of_time, 0.001), 
            r, q, params
        )
        option_prices.append(option_price)
        
        deltas.append(sim_env.current_delta)
        hedge_positions.append(sim_env.current_count_stocks)
        hedge_errors.append(sim_env.hedge_error)
        portfolio_values.append(sim_env.cash + 
                              sim_env.current_price_stock * sim_env.current_count_stocks + 
                              option_price)
        
        # Агент выбирает действие (0-100)
        action = agent.act(state)
        actions_history.append(action)
        actions_percent.append(action)  # 0-100 = процент хеджа
        
        state, reward, done = sim_env.step(action)
        day += 1
    
    print(f"✅ Симуляция завершена! Дней: {day}")
    
    # ============= 4. ВИЗУАЛИЗАЦИЯ =============
    print("\n🎨 ЭТАП 3: СОЗДАНИЕ ГРАФИКОВ")
    print("-"*50)
    
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # [1] Цена акции и опциона
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('📈 Цена акции и опциона', fontsize=14, fontweight='bold', pad=15)
    ax1.plot(days, stock_prices, 'b-', linewidth=2, label='Цена акции (S)')
    ax1.set_xlabel('День')
    ax1.set_ylabel('Цена акции', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax1b = ax1.twinx()
    ax1b.plot(days, option_prices, 'r-', linewidth=2, label='Цена опциона')
    ax1b.set_ylabel('Цена опциона', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper left')
    
    # [2] Дельта и позиция хеджа
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('🛡️ Дельта опциона vs Позиция хеджа', fontsize=14, fontweight='bold', pad=15)
    ax2.plot(days, deltas, 'g-', linewidth=2, label='Дельта опциона')
    ax2.plot(days, hedge_positions, 'orange', linewidth=2, linestyle='--', 
             label='Позиция в акциях')
    ax2.set_xlabel('День')
    ax2.set_ylabel('Значение')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # [3] Ошибка хеджирования
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('⚠️ Ошибка хеджирования', fontsize=12, fontweight='bold')
    ax3.plot(days, hedge_errors, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('День')
    ax3.set_ylabel('Hedge Error')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    # [4] Действия агента (101 действие!)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_title('🎮 Действия агента (% хеджа)', fontsize=12, fontweight='bold')
    
    # График процента хеджа по дням
    days_actions = days[:-1] if len(days) == len(actions_percent) else days[:len(actions_percent)]
    ax4.plot(days_actions, actions_percent, 'b-', linewidth=2, marker='o', markersize=4)
    ax4.fill_between(days_actions, 0, actions_percent, alpha=0.3, color='blue')
    ax4.set_xlabel('День')
    ax4.set_ylabel('Процент хеджа (%)')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    
    # [5] Распределение действий (гистограмма)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_title('📊 Распределение действий', fontsize=12, fontweight='bold')
    
    # Группируем по диапазонам: 0%, 1-25%, 26-50%, 51-75%, 76-99%, 100%
    bins = [0, 1, 25, 50, 75, 99, 100]
    bin_labels = ['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']
    
    hist, _ = np.histogram(actions_percent, bins=bins)
    colors = plt.cm.RdYlGn(hist / max(hist) if max(hist) > 0 else hist)
    
    bars = ax5.bar(bin_labels, hist, color=colors, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Процент хеджа')
    ax5.set_ylabel('Частота')
    ax5.tick_params(axis='x', rotation=45)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars, hist):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # [6] Стоимость портфеля
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_title('💰 Стоимость портфеля', fontsize=14, fontweight='bold', pad=15)
    ax6.plot(days, portfolio_values, 'b-', linewidth=2, label='Портфель')
    
    # Линейный тренд
    if len(days) > 1:
        z = np.polyfit(days, portfolio_values, 1)
        p = np.poly1d(z)
        ax6.plot(days, p(days), 'r--', linewidth=1.5, 
                label=f'Тренд: {z[0]:.2f} $/день')
    
    ax6.set_xlabel('День')
    ax6.set_ylabel('Стоимость ($)')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # Итоговые метрики
    optimal_hedge = np.mean(actions_percent) if actions_percent else 0
    
    fig.text(0.02, 0.02, 
             f"📊 ИТОГОВЫЕ МЕТРИКИ (101 действие):\n"
             f"• Начальная цена акции: ${S0:.1f}\n"
             f"• Страйк: ${K:.1f}\n"
             f"• Время до экспирации: {T*365:.0f} дней\n"
             f"• Финальная цена акции: ${stock_prices[-1]:.2f}\n"
             f"• Финальная цена опциона: ${option_prices[-1]:.2f}\n"
             f"• Финальная ошибка хеджа: {hedge_errors[-1]:.4f}\n"
             f"• Финальная стоимость портфеля: ${portfolio_values[-1]:.2f}\n"
             f"• Средняя ошибка хеджа: {np.mean(np.abs(hedge_errors)):.4f}\n"
             f"• Средний процент хеджа: {optimal_hedge:.1f}%\n"
             f"• Лучшая награда при обучении: {best_reward:.2f}",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    plt.suptitle('🤖 DQN АГЕНТ: ДИНАМИЧЕСКОЕ ХЕДЖИРОВАНИЕ (101 ДЕЙСТВИЕ, HESTON)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('dqn_hedging_101_actions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Графики сохранены в 'dqn_hedging_101_actions.png'")
    print(f"📊 Средний процент хеджа: {optimal_hedge:.1f}%")
    print(f"🏆 Лучшая награда: {best_reward:.2f}")
    
    agent.save("trained_agent_101.pth")
    print("✅ Модель сохранена в 'trained_agent_101.pth'")
    
    return agent, sim_env

if __name__ == "__main__":
    agent, env = train_and_visualize()
    
    print("\n" + "="*70)
    print("🎯 ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("="*70)
