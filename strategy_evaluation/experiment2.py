import pandas as pd
import matplotlib.pyplot as plt
import StrategyLearner as sl
import datetime as dt

def impact_experiment():
    impacts = [0.0, 0.005, 0.01]
    results = {}
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    for impact in impacts:
        learner = sl.StrategyLearner(impact=impact)
        learner.add_evidence(symbol="JPM", sd=sd, ed=ed)
        df_trades = learner.testPolicy(symbol="JPM", sd=sd, ed=ed)
        # TODO: Calculate portfolio values
        results[impact] = df_trades  # Placeholder for actual results

    # Plotting the results
    plt.figure(figsize=(10, 5))
    for impact, trades in results.items():
        # plt.plot(portfolio_values[impact], label=f'Impact: {impact}')
        pass
    plt.title('Impact of different levels on trading performance')
    plt.legend()
    plt.show()

def author():
    return 'pamin33'


if __name__ == "__main__":
    impact_experiment()




