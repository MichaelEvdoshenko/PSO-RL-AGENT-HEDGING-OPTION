import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RL_agent.RL_AGENT_AND_ENVIRONMENT import HedgingEnv, AgentDQN
from RL_agent.hekston_model import compute_price_call_single, calculate_heston_delta
from collections import deque

def test(params, S0, K, T, r, q, agent):
    sim_env = HedgingEnv(S0=S0, T=T, K=K, q=q, r=r, params_option=params)
    state = sim_env.reset()

    days = []
    stock_prices = []
    option_prices = []
    deltas = []
    hedge_positions = []
    actions_history = []
    actions_percent = []
    portfolio_values = []
    hedge_errors = []
    
    day = 0
    done = False
    
    print(f"начало симуляций с T = {T}, S0 = {S0}, r = {r}, q = {q}, K = {K}")
    
    while not done:
        days.append(day)
        stock_prices.append(sim_env.current_price_stock)
        option_price = compute_price_call_single(
            sim_env.current_price_stock, K, 
            max(sim_env.rest_of_time, 0.001), 
            r, q, params
        )
        option_price += np.random.normal(0, 0.01 * option_price)
        option_prices.append(option_price)
        
        deltas.append(sim_env.current_delta)
        hedge_positions.append(sim_env.current_count_stocks)
        hedge_errors.append(sim_env.hedge_error)
        portfolio_values.append(sim_env.cash + 
                              sim_env.current_price_stock * sim_env.current_count_stocks + 
                              option_price)
        
        action = agent.act(state)
        actions_history.append(action)
        actions_percent.append(action)
     
        state, reward, done = sim_env.step(action)
        day += 1
    
    print("конец симуляции")
    print("создание графиков")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Цена акции и опциона', fontsize=12, pad=10)
    ax1.plot(days, stock_prices, 'b-', linewidth=2, label='Акция')
    ax1.set_xlabel('День')
    ax1.set_ylabel('Цена акции', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(days, option_prices, 'r-', linewidth=2, label='Опцион')
    ax1b.set_ylabel('Цена опциона', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper left')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Ошибка хеджирования', fontsize=12, pad=10)
    ax2.plot(days, hedge_errors, 'purple', linewidth=2)
    ax2.set_xlabel('День')
    ax2.set_ylabel('Ошибка')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)


    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Действия агента (% хеджа)', fontsize=12, pad=10)
    ax3.plot(days, actions_percent, 'b-', linewidth=2)
    ax3.fill_between(days, 0, actions_percent, alpha=0.2, color='blue')
    ax3.set_xlabel('День')
    ax3.set_ylabel('Процент хеджа')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)


    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title('Распределение действий', fontsize=12, pad=10)
    bins = [0, 1, 25, 50, 75, 99, 100]
    bin_labels = ['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']
    hist, _ = np.histogram(actions_percent, bins=bins)
    bars = ax4.bar(bin_labels, hist, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Процент хеджа')
    ax4.set_ylabel('Частота')
    ax4.tick_params(axis='x', rotation=45)

    for bar, count in zip(bars, hist):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{count}', ha='center', va='bottom', fontsize=8)


    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title('Стоимость портфеля', fontsize=12, pad=10)
    ax5.plot(days, portfolio_values, 'b-', linewidth=2)
    ax5.set_xlabel('День')
    ax5.set_ylabel('Стоимость ($)')
    ax5.grid(True, alpha=0.3)

    optimal_hedge = np.mean(actions_percent) if actions_percent else 0
    metrics_text = (f"Начальная цена: ${S0:.1f}\n"
                    f"Страйк: ${K:.1f}\n"
                    f"Дней: {T*365:.0f}\n"
                    f"Финальная цена: ${stock_prices[-1]:.2f}\n"
                    f"Финальная цена опциона: ${option_prices[-1]:.2f}\n"
                    f"Финальная ошибка: {hedge_errors[-1]:.4f}\n"
                    f"Средний % хеджа: {optimal_hedge:.1f}%\n"
                    f"Финальная стоимость портфеля: ${portfolio_values[-1]:.2f}"
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.suptitle('DQN агент - хеджирование опциона', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig('dqn_hedging_results.png', dpi=120, bbox_inches='tight')
    plt.show()

    print("график сохранен в 'dqn_hedging_results.png'")
    return agent, sim_env
