import pandas as pd
import numpy as np


def compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values.

    Parameters:
        trades (DataFrame): Trades for each day; a non-zero value indicates a transaction.
        start_val (int): The starting cash in dollars.
        commission (float): Fixed commission for each trade.
        impact (float): Market impact of each trade, a multiplier effect on price.

    Returns:
        portvals (DataFrame): Portfolio value for each trading day.
    """
    if trades.empty:
        return pd.DataFrame({'port_val': []})

    # Get a list of all symbols traded
    symbols = trades.columns.tolist()

    # Assume we have a 'prices' DataFrame already available that matches 'trades' dates
    # For now, let's simulate 'prices' as constant (you need to use actual adjusted closing prices)
    dates = trades.index
    prices = pd.DataFrame(index=dates, data={symbol: np.random.uniform(50, 150, size=len(dates)) for symbol in symbols})

    # Including cash in our portfolio
    trades['Cash'] = 0.0

    # Calculate the impact on price due to each trade and adjust cash accordingly
    for symbol in symbols:
        # Prices are adjusted by the impact and the direction of the trade
        price_impact = prices[symbol] * trades[symbol] * impact
        prices[symbol] += price_impact

        # Update cash to reflect buys/sells
        trades['Cash'] -= (prices[symbol] * trades[symbol]) + (commission * np.sign(trades[symbol]))

    # Include the initial cash start value
    trades['Cash'].iloc[0] += start_val

    # Calculate holdings of each symbol (cumulative sum of trades)
    holdings = trades[symbols].cumsum()

    # Calculate holdings in dollars
    holdings = holdings * prices[symbols]

    # Sum all holdings and cash to get total portfolio value
    portvals = holdings.sum(axis=1) + trades['Cash'].cumsum()

    # Return as DataFrame
    return pd.DataFrame(portvals, columns=['port_val'])


def author():
    return 'pamin33'


# Example usage:
if __name__ == "__main__":
    # Example: DataFrame with trade data
    dates = pd.date_range('2020-01-01', '2020-01-10')
    data = {'AAPL': [1000, 0, -1000, 0, 1000, 0, -1000, 0, 1000, -1000]}
    trades = pd.DataFrame(index=dates, data=data)

    # Compute the portfolio values
    portvals = compute_portvals(trades)
    print(portvals)

