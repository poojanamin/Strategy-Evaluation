import pandas as pd
import util as ut
import datetime as dt


class BenchmarkStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def addEvidence(self, symbol=0, sd=0, ed=0, sv=0):
        """Keep this so that API is valid."""
        pass

    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):
        """Benchmark is to buy 1000 shares and hold."""
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)  # automatically adds SPY

        orders = pd.DataFrame(index=prices.index)
        orders["Symbol"] = symbol
        orders["Order"] = ""
        orders["Shares"] = 0
        orders.iloc[0] = [symbol, "BUY", 1000]
        orders.iloc[-1] = [symbol, "SELL", -1000]

        if self.verbose:
            print(type(orders))  # it better be a DataFrame!
            print(orders)
        return orders
