import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn 
import random
from collections import deque
from simulate_hekston import compute_price_call_single, calculate_heston_gamma, calculate_heston_delta
from gymnasium import spaces

class HedgingEnv():
    def __init__(self, S0, T, K, q, r, params_option):
        '''
        S0 - цена на рисковый актив в момент времени 0
        T - время в годах до экспирации 
        K - страйк опциона колл
        q - девидендная доходность
        r - безрисковая ставка (считаем const на протяжении всего периода)
        params_option = {v0,kappa,theta,xi,cor} см. simulate_hekston.py
        '''
        self.T = T
        self.rest_of_time = T 
        self.S0 = S0
        self.current_price_stock = self.S0
        self.current_price_option = compute_price_call_single(S0, K, T, r, q, params_option, n_paths=5000)
        self.current_count_stocks = 0
        self.current_count_options = 1
        self.portfolio_values = self.current_price_option * self.current_count_options + self.current_price_stock * self.current_count_stocks
        self.r = r
        self.q = q
        self.K = K
        self.options_params = params_option

        self.initial_capital = 10000.0
        self.cash = self.initial_capital
        self.transaction_cost = 0.001
        self.daily_r = self.r / 365

        self.current_day = 0
        self.total_days = int(T * 365)
        self.days_passed = 0
    
        self.sigma_historical = 0.25
        self.price_history = [S0]
    
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.current_vega = 0.0
        self.hedge_error = 0.0
    
        self.max_steps = self.total_days
        self.step_count = 0
    
        self.action_space = spaces.Discrete(101)
        self.observation_space = self.define_observation_space()
    
        self.history = {
            'prices': [],
            'deltas': [],
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'cash_history': [],
            'hedge_errors': []
        }

        self.dt = 1/365
        self.reward_scale = 1.0
        self.include_volatility = True

    def define_observation_space(self):

        low = np.array([
            0.5,      # moneyness (S/K)
            0.0,      # time to expiry (нормализованное: [0,1])
            0.0,      # delta опциона (для call: [0,1])
            -5.0,     # текущая позиция в акциях
            0.0,      # cash (нормализованный)
            -1.0,     # предыдущее действие
        ], dtype=np.float32)

        high = np.array([
            1.5,
            1.0,
            1.0,
            5.0,
            2.0,
            1.0,
        ], dtype=np.float32)

        return spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def reset(self):
        self.rest_of_time = self.T
        self.current_price_stock = self.S0
        self.current_count_stocks = 0
        self.current_count_options = 1

        self.initial_capital = 10000.0
        self.cash = self.initial_capital

        self.current_day = 0
        self.total_days = int(self.T * 365)
        self.days_passed = 0

        self.price_history = [self.S0]
    
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.current_vega = 0.0
        self.hedge_error = 0.0

        self.max_steps = self.total_days
        self.step_count = 0
        
        return self.get_state()
    
    def get_state(self):
        moneyness = self.current_price_stock / self.K
    
        time_left = max(self.rest_of_time, 0) / self.T
    
        hedge_position = self.current_count_stocks
    
        cash_ratio = self.cash / self.initial_capital

        if self.history["actions"]:
            prev_action = self.history["actions"][-1] 
        else:
            prev_action = 0.0
    
        return np.array([
            moneyness,
            time_left,
            self.current_delta,
            hedge_position,
            cash_ratio,
            prev_action
        ], dtype=np.float32)

    def compute_portfolio_price(self):

        stock_value = self.current_price_stock * self.current_count_stocks

        time_to_expiry = max(self.rest_of_time, 0.001)
        current_option_price = compute_price_call_single(self.current_price_stock, self.K, time_to_expiry, self.r, self.q, self.options_params)
        option_value = current_option_price * self.current_count_options

        portfolio_value = self.cash + stock_value + option_value
    
        return portfolio_value

    def trading(self, move):
        if (move == 0):
            return 0.0
    
        count_trade = self.hedge_error * (move/100) 


        if abs(count_trade) < 0.001:
            return 0.0

        transaction_cost = abs(count_trade) * self.current_price_stock * self.transaction_cost
    
        if count_trade < 0:
            # Покупка
            money_for_trade = abs(count_trade) * self.current_price_stock
            self.cash -= money_for_trade + transaction_cost
            self.current_count_stocks += abs(count_trade)
        else:  # count_trade > 0
            # Продажа  
            money_for_trade = abs(count_trade) * self.current_price_stock
            self.cash += money_for_trade - transaction_cost  # ← Получаем деньги за вычетом комиссии
            self.current_count_stocks -= abs(count_trade)
    
        return transaction_cost

    def calculate_pnl_volatility(self, window=10):
        """
        Расчет волатильности портфеля за скользящее окно
        window: размер окна для расчета (последние N дней)
        Returns: volatility - стандартное отклонение P&L
        """
        portfolio_values = self.history['portfolio_values']
    
        if len(portfolio_values) < window + 1:
            return 0.0
    
        recent_values = portfolio_values[-(window + 1):]
    
        daily_pnls = []
        for i in range(1, len(recent_values)):
            daily_pnl = recent_values[i] - recent_values[i-1]
            daily_pnls.append(daily_pnl)
    
        if len(daily_pnls) <= 1:
            return 0.0
    
        volatility = np.std(daily_pnls)
    
        return volatility

    def calculate_reward(self, daily_pnl, transaction_cost):

    # ============= 1. ПРИБЫЛЬ (ОСНОВА) =============
    # В реальной жизни мы хотим ЗАРАБАТЫВАТЬ!
        profit_reward = daily_pnl * 1.0  # Каждый доллар прибыли = +1
    
    
    # ============= 2. ШТРАФ ЗА РИСК (НЕ ХЕДЖ) =============
    # 100% хедж = 0 риск = нет штрафа
    # 0% хедж = большой риск = большой штраф
        risk_penalty = -abs(self.hedge_error) * 200.0  # Чем больше ошибка, тем хуже
    
    
    # ============= 3. TRANSACTION COSTS (ПРОТИВ 100% ХЕДЖА) =============
    # 100% хедж = ПОСТОЯННЫЕ сделки = КРУПНЫЙ штраф
    # 0% хедж = нет сделок = нет штрафа
        cost_penalty = -transaction_cost * 500.0  # Сильный штраф за торговлю!
    
    # 🔥 ИМЕННО ЭТО ЗАСТАВИТ АГЕНТА НЕ ХЕДЖИРОВАТЬ 100%!
    # Теперь 100% хедж = маленький risk_penalty, но ОГРОМНЫЙ cost_penalty
    # Агент должен найти БАЛАНС
    
    
    # ============= 4. SLIPPAGE (РЕАЛЬНЫЙ РЫНОК) =============
    # Чем больше сделка, тем хуже цена исполнения
        if transaction_cost > 0:
            # Примерно оцениваем размер сделки по transaction_cost
            trade_size = transaction_cost / (self.current_price_stock * self.transaction_cost)
            slippage_penalty = -abs(trade_size) * 10.0  # Штраф за крупные сделки
        else:
            slippage_penalty = 0.0
    
    
    # ============= 5. MARGIN REQUIREMENTS (КАПИТАЛ) =============
    # Нельзя использовать весь капитал на маржинальные требования
    # Чем больше позиция, тем больше заморожено денег
        margin_penalty = -abs(self.current_count_stocks) * 2.0
    
    
    # ============= 6. РЕЖИМЫ РЫНКА =============
        if len(self.history['portfolio_values']) > 5:
            recent_returns = np.diff(self.history['portfolio_values'][-5:])
            volatility = np.std(recent_returns)
        
        # В спокойном рынке - меньше хеджируем, экономим costs
        # В волатильном рынке - больше хеджируем, защищаемся
            if volatility > 100:  # Высокая волатильность
                # Штрафуем за маленький хедж
                adaptive_penalty = -abs(self.hedge_error) * 300.0
            else:  # Низкая волатильность
            # Штрафуем за большой хедж (лишние costs)
                adaptive_penalty = abs(self.hedge_error) * -100.0 + abs(transaction_cost) * -200.0
        else:
            adaptive_penalty = 0.0
    
    
    # ============= 7. CRASH PROTECTION =============
        if daily_pnl < -100:  # Крупный убыток
            crash_penalty = -abs(daily_pnl) * 3.0
            print(f"⚠️ CRASH! PnL: {daily_pnl:.2f}, Penalty: {crash_penalty:.2f}")
        else:
            crash_penalty = 0.0
    
    
    # ============= 8. БОНУС ЗА СТАБИЛЬНОСТЬ =============
    # Sharpe ratio-like: награждаем за МАЛУЮ волатильность PnL
        if len(self.history['portfolio_values']) > 10:
            pnl_history = np.diff(self.history['portfolio_values'][-10:])
            if np.std(pnl_history) > 0:
                sharpe_bonus = (np.mean(pnl_history) / np.std(pnl_history)) * 10.0
            else:
                sharpe_bonus = 0.0
        else:
            sharpe_bonus = 0.0
    
    
    # ============= СУММИРУЕМ =============
        total_reward = (
            profit_reward +                 # Хотим прибыль
            risk_penalty +                 # Не хотим риск
            cost_penalty +                # Не хотим платить комиссии
            slippage_penalty +           # Не хотим большие сделки
            margin_penalty +            # Не хотим занимать много капитала
            adaptive_penalty +         # Адаптируемся к рынку
            crash_penalty +           # Защита от катастроф
            sharpe_bonus            # Хотим стабильность
        )
    
        return total_reward

    def step(self, move):
        "move принимает одно из значений: [0, 1, 2] = {хеджировать на 0%, хеджировать на 50%, хеджировать на 100%}"
        
        self.history["actions"].append(move)
        self.prev_action = move

        self.old_portfolio_price = self.compute_portfolio_price()

        self.current_day += 1
        self.days_passed += 1
        self.rest_of_time = max(0, self.rest_of_time - self.dt)

        shock = np.random.normal(0, self.sigma_historical * np.sqrt(self.dt))
        self.current_price_stock *= np.exp(shock)
        self.price_history.append(self.current_price_stock)

        self.current_delta = calculate_heston_delta(self.current_price_stock, self.K, self.rest_of_time, self.r, self.q, self.options_params)
        self.current_gamma = calculate_heston_gamma(self.current_price_stock, self.K, self.rest_of_time, self.r, self.q, self.options_params)
        target_hedge = -self.current_delta * self.current_count_options
        self.hedge_error = self.current_count_stocks - target_hedge

        transaction_price = self.trading(move)

        self.hedge_error = self.current_count_stocks - target_hedge
        new_portfolio_price = self.compute_portfolio_price()
        daily_pnl = new_portfolio_price - self.old_portfolio_price
        reward = self.calculate_reward(daily_pnl, transaction_price)
        
        if (self.current_day >= self.total_days):
            done = True
        else:
            done = False

        self.history['prices'].append(self.current_price_stock)
        self.history['deltas'].append(self.current_delta)
        self.history['rewards'].append(reward)
        self.history['portfolio_values'].append(new_portfolio_price)
        self.history['cash_history'].append(self.cash)
        self.history['hedge_errors'].append(self.hedge_error)

        return  self.get_state(), reward, done


class AgentDQN:
    def __init__(self, state_dim=6, action_dim=101):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = self.build_network().to(self.device)
        self.target_net = self.build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.01

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=2000)
        self.update_target_every = 4
        self.train_step = 0

        self.batch_size = 50
    
    def build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return torch.argmax(q_values).item()
    
    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="dqn_agent.pth"):
        """Сохранение модели"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'memory_size': len(self.memory)
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename="dqn_agent.pth"):
        """Загрузка модели"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"Model loaded from {filename}")
        print(f"Epsilon: {self.epsilon}, Train step: {self.train_step}")