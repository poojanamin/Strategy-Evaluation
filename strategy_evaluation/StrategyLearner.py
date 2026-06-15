"""
Implementing a Machine Learning and Q-Learning based Strategy Learner for Trading

@Name : Pooja Amin
@UserID : pamin33
"""

import numpy as np
import numpy as np
import pandas as pd
import datetime as dt


class QLearner:
    def __init__(self, num_states=100, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.random.uniform(-1, 1, size=(num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))

    def query_set_state(self, s):
        self.s = s

    def query(self, s_prime, r):
        if np.random.uniform() <= self.rar:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.Q[s_prime])

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
                    r + self.gamma * self.Q[s_prime, action])

        if self.dyna > 0:
            self.T[self.s, self.a, s_prime] += 1
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            for _ in range(self.dyna):
                dyna_s = np.random.randint(0, self.num_states)
                dyna_a = np.random.randint(0, self.num_actions)
                dyna_s_prime = np.argmax(self.T[dyna_s, dyna_a])
                dyna_r = self.R[dyna_s, dyna_a]
                self.Q[dyna_s, dyna_a] = (1 - self.alpha) * self.Q[dyna_s, dyna_a] + self.alpha * (
                            dyna_r + self.gamma * self.Q[dyna_s_prime, np.argmax(self.Q[dyna_s_prime])])

        self.rar *= self.radr
        self.s = s_prime
        self.a = action
        return action


class StrategyLearner:
    def __init__(self, impact=0.0, commission=9.95, num_states=1000, num_actions=3, **kwargs):
        self.impact = impact
        self.commission = commission
        self.num_states = num_states
        self.num_actions = num_actions
        self.verbose = kwargs.get('verbose', False)  # Handle verbose argument
        self.threshold = 0.0
        self.learner = QLearner(num_states=num_states, num_actions=num_actions)

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Trains the learner with historical data.
        """
        # Fetch training data
        dates = pd.date_range(sd, ed)
        prices = self.get_data(symbol, dates)
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        bollinger_upper = sma + (2 * std)
        bollinger_lower = sma - (2 * std)
        rsi = self.calculate_rsi(prices)

        # Create features
        X = pd.DataFrame({
            'price': prices,
            'sma': sma,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower,
            'rsi': rsi
        })
        X.fillna(method='bfill', inplace=True)

        # Train the learner
        for i in range(X.shape[0] - 1):
            state = i % self.num_states
            self.learner.query_set_state(state)
            action = self.learner.query(i + 1, 0)

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
        Tests the learner using data outside the training data.
        """
        # Fetch test data
        dates = pd.date_range(sd, ed)
        prices = self.get_data(symbol, dates)
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        bollinger_upper = sma + (2 * std)
        bollinger_lower = sma - (2 * std)
        rsi = self.calculate_rsi(prices)

        # Create features
        X = pd.DataFrame({
            'price': prices,
            'sma': sma,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower,
            'rsi': rsi
        })
        X.fillna(method='bfill', inplace=True)

        # Predict
        preds = []
        state = 0
        for i in range(X.shape[0] - 1):
            self.learner.query_set_state(state)
            action = self.learner.query(i + 1, 0)
            preds.append(action)
            state = (state + 1) % self.num_states

        trades = pd.DataFrame(data=preds * 1000, index=prices.index[:-1], columns=[symbol])

        return trades

    def get_data(self, symbol, dates):
        """
        Fetch stock data (Adjusted Close) for given symbols from CSV files.
        """
        df = pd.read_csv("data/{}.csv".format(symbol), index_col='Date',
                         parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df = df.rename(columns={'Adj Close': symbol})
        df = df.reindex(dates)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df[symbol]

    def calculate_rsi(self, prices, window=14):
        """
        Calculate the Relative Strength Index (RSI).
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def author():
    return 'pamin33'


# Example usage
if __name__ == "__main__":
    sl = StrategyLearner()
    sl.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades = sl.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    print(trades)
