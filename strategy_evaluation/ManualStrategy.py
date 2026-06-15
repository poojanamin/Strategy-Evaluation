"""
Manual Trading Strategy Implementation Using RSI, MACD, and Bollinger Bands

@Name : Pooja Amin
@UserID : pamin33

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from marketsimcode import compute_portvals

class ManualStrategy:
    def __init__(self, symbol="JPM", start_val=100000, commission=9.95, impact=0.005):
        self.symbol = symbol
        self.start_val = start_val
        self.commission = commission
        self.impact = impact

    def testPolicy(self, symbol, sd, ed, sv):
        dates = pd.date_range(sd, ed)
        prices = self.get_data(symbol, dates)  # Get price data

        # Calculate indicators
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        bollinger_upper = sma + (2 * std)
        bollinger_lower = sma - (2 * std)
        rsi = self.calculate_rsi(prices)

        trades = pd.DataFrame(data=np.zeros(prices.shape), index=prices.index, columns=[symbol])
        holdings = 0

        # Trading signals based on the indicators
        for i in range(1, len(prices)):
            # Entry signal for long position
            if prices.iloc[i] < bollinger_lower.iloc[i] and rsi.iloc[i] < 30 and prices.iloc[i] > sma.iloc[i]:
                if holdings == 0:  # Not in the market
                    trades.iloc[i] = 1000
                    holdings = 1000
                elif holdings == -1000:  # In the market short
                    trades.iloc[i] = 2000
                    holdings = 1000

            # Entry signal for short position
            elif prices.iloc[i] > bollinger_upper.iloc[i] and rsi.iloc[i] > 70 and prices.iloc[i] < sma.iloc[i]:
                if holdings == 0:  # Not in the market
                    trades.iloc[i] = -1000
                    holdings = -1000
                elif holdings == 1000:  # In the market long
                    trades.iloc[i] = -2000
                    holdings = -1000

            # Exit signal
            elif holdings != 0 and ((holdings == 1000 and prices.iloc[i] < sma.iloc[i]) or (holdings == -1000 and prices.iloc[i] > sma.iloc[i])):
                trades.iloc[i] = -holdings
                holdings = 0

        return trades

    def get_data(self, symbol, dates):
        """Fetch stock data (Adjusted Close) for given symbols from CSV files."""
        df = pd.read_csv("data/{}.csv".format(symbol), index_col='Date',
                         parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df = df.rename(columns={'Adj Close': symbol})
        df = df.reindex(dates)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df[symbol]

    def calculate_rsi(self, prices, window=14):
        """Calculate the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot_results(self, trades):
        """Plot the results of the strategy."""
        prices = self.get_data(self.symbol, trades.index)
        portvals = compute_portvals(trades, start_val=self.start_val, commission=self.commission, impact=self.impact)
        portvals /= portvals.iloc[0]

        plt.figure(figsize=(10, 6))
        plt.plot(portvals, label='Strategy', color='red')
        plt.title('Manual Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Normalized Portfolio Value')
        plt.legend()
        plt.show()


def author():
    return 'pamin33'


# Example usage
if __name__ == "__main__":
    ms = ManualStrategy()
    trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    ms.plot_results(trades)

