import datetime as dt
import pandas as pd
import util
import indicators
from qlearning_robot.QLearner import QLearner as Learner
from dataclasses import dataclass

@dataclass
class Holding:
    cash: int
    shares: int
    equity: int


class QLearner(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0, testing=False, n_bins=5):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.testing = testing  # Decides which type of order df to return.
        self.indicators = ['macd_diff', 'rsi', 'price_sma_8']
        self.n_bins = n_bins
        self.bins = {}
        self.num_states = self.get_num_states()
        self.num_actions = 3  # buy, sell, hold
        self.learner = Learner(self.num_states, self.num_actions)

    def row_to_state(self, holding, df_row):
        """Transforms a row into a state value."""
        holding = (holding + 1000) // 1000
        assert(holding in [0, 1, 2])

        # For each indicator that goes into the state the interval becomes
        # smaller based on how many bins the indicator has.  The first
        # 'indicator' is the information about how many shares we are currently
        # holding. So for example, if I have 450 states then the intervall (aka
        # remaining_states) is 150 because there are three values for holding:
        #   holding = 0 -> state = 0 * 150 = 0
        #   holding = 1 -> state = 1 * 150 = 150
        #   holding = 2 -> state = 2 * 150 = 300
        remaining_states = self.num_states // 3
        state = holding * remaining_states

        for indicator in self.indicators:
            value = df_row[indicator]
            bin_n = self.indicator_value_to_bin(indicator, value)
            remaining_states //= self.n_bins
            state += bin_n * remaining_states
        return state

    def indicator_value_to_bin(self, indicator, value):
        for i, upper_bound in enumerate(self.bins[indicator]):
            if value < upper_bound:
                return i
        return i + 1

    def add_indicators(self, df, symbol):
        """Add indicators for learning to DataFrame."""
        for indicator in self.indicators:
            if indicator == "macd_diff":
                indicators.macd(df, symbol)
                df.drop(columns=["macd", "macd_signal"], inplace=True)
            elif indicator == "rsi":
                indicators.rsi(df, symbol)
            elif indicator.startswith("price_sma_"):
                period = int(indicator.replace("price_sma_", ""))
                indicators.price_sma(df, symbol, [period])
        df.drop(columns=["SPY"], inplace=True)
        df.dropna(inplace=True)

    def bin_indicators(self, df):
        """Create bins for indicators."""
        for indicator in self.indicators:
            ser, bins = pd.qcut(df[indicator], self.n_bins, retbins=True)
            self.bins[indicator] = bins[1:self.n_bins]

    def get_num_states(self):
        """Return the total num of states."""
        num_states = 3  # Three states holding (1000, 0, -1000)
        for _ in self.indicators:
            num_states *= self.n_bins
        return num_states

    def handle_order(self, action, holding, adj_closing_price):
        shares = 0
        if action == 0:  # buy
            if holding.shares == 0 or holding.shares == -1000:
                shares = 1000
        elif action == 1:  # sell
            if holding.shares== 0 or holding.shares == 1000:
                shares = -1000
        elif action == 2:  # hold
            shares = 0

        cost = shares * adj_closing_price
        if shares != 0:
            # Charge commission and deduct impact penalty
            holding.cash -= self.commission
            holding.cash -= (self.impact * adj_closing_price * abs(shares))
            holding.cash -= cost
            holding.shares += shares

        holding.equity = holding.cash + holding.shares * adj_closing_price

    def get_reward(self, equity, new_equity):
        if new_equity > equity:
            return 1
        return -1

    def train(self, df, symbol, sv):
        holding = Holding(sv, 0, sv)

        row = df.iloc[0]
        state = self.row_to_state(holding.shares, row)
        action = self.learner.querysetstate(state)
        adj_closing_price = row[symbol]
        equity = holding.equity
        self.handle_order(action, holding, adj_closing_price)

        for index, row in df.iloc[1:].iterrows():
            adj_closing_price = row[symbol]
            new_equity = holding.cash + holding.shares * adj_closing_price
            r = self.get_reward(equity, new_equity)
            s_prime = self.row_to_state(holding.shares, row)
            a = self.learner.query(s_prime, r)
            equity = new_equity
            self.handle_order(a, holding, adj_closing_price)
            if self.verbose:
                print(f"{holding=} {s_prime=} {r=} {a=}")

    def addEvidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):
        df = util.get_data([symbol], pd.date_range(sd, ed))
        self.add_indicators(df, symbol)
        self.bin_indicators(df)

        for _ in range(15):
            self.train(df, symbol, sv)

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):
        df = util.get_data([symbol], pd.date_range(sd, ed))
        orders = pd.DataFrame(index=df.index)
        orders["Symbol"] = symbol
        orders["Order"] = ""
        orders["Shares"] = 0
        shares = orders["Shares"]
        self.add_indicators(df, symbol)
        holding = 0

        for index, row in df.iterrows():
            state = self.row_to_state(holding, row)
            action = self.learner.querysetstate(state)

            if action == 0:  # buy
                if holding == 0 or holding == -1000:
                    holding += 1000
                    orders.loc[index, "Shares"] = 1000
            elif action == 1:  # sell
                if holding == 0 or holding == 1000:
                    holding -= 1000
                    orders.loc[index, "Shares"] = -1000
            elif action == 2:  # hold
                pass

        if self.testing:
            return orders
        else:
            return orders[["Shares"]]
