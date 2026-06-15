import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals


def experiment1():
    # Parameters
    symbol = "JPM"
    start_val = 100000
    commission = 9.95
    impact = 0.005
    sd_insample = dt.datetime(2008, 1, 1)
    ed_insample = dt.datetime(2009, 12, 31)
    sd_outsample = dt.datetime(2010, 1, 1)
    ed_outsample = dt.datetime(2011, 12, 31)

    # Instantiate strategies
    ms = ManualStrategy()
    sl = StrategyLearner(impact=impact)

    # Training the strategy learner
    sl.add_evidence(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=start_val)

    # Testing both strategies
    df_trades_manual_insample = ms.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=start_val)
    df_trades_strategy_insample = sl.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=start_val)
    df_trades_manual_outsample = ms.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=start_val)
    df_trades_strategy_outsample = sl.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=start_val)

    # Compute portfolio values
    portvals_manual_insample = compute_portvals(df_trades_manual_insample, start_val, commission, impact)
    portvals_strategy_insample = compute_portvals(df_trades_strategy_insample, start_val, commission, impact)
    portvals_manual_outsample = compute_portvals(df_trades_manual_outsample, start_val, commission, impact)
    portvals_strategy_outsample = compute_portvals(df_trades_strategy_outsample, start_val, commission, impact)

    # Normalize the portfolio values to 1 at the start of the period for comparison
    portvals_manual_insample /= portvals_manual_insample.iloc[0]
    portvals_strategy_insample /= portvals_strategy_insample.iloc[0]
    portvals_manual_outsample /= portvals_manual_outsample.iloc[0]
    portvals_strategy_outsample /= portvals_strategy_outsample.iloc[0]

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(portvals_manual_insample, 'r', label='Manual Strategy In-Sample')
    plt.plot(portvals_strategy_insample, 'b', label='Strategy Learner In-Sample')
    plt.title('In-Sample Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(portvals_manual_outsample, 'r', label='Manual Strategy Out-of-Sample')
    plt.plot(portvals_strategy_outsample, 'b', label='Strategy Learner Out-of-Sample')
    plt.title('Out-of-Sample Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def author():
    return 'pamin33'


if __name__ == "__main__":
    experiment1()





