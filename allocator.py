import numpy as np
import pandas as pd
import riskfolio as rp

class Allocator:
    def __init__(self):
        pass

    def mean_variance_optimization(self, data, min_weight=0.3, take_pct_change=True):
        if take_pct_change:
            data = data.pct_change().dropna()

        port = rp.Portfolio(data)

        method_mu = 'hist'  # Method to estimate expected returns based on historical data.
        method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

        model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
        rm = 'MV'  # Risk measure used, this time will be variance
        obj = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
        hist = True  # Use historical scenarios for risk measures that depend on scenarios
        rf = 0  # Risk free rate
        l = 0  # Risk aversion factor, only useful when obj is 'Utility'

        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        w.loc[w['weights'] < min_weight, 'weights'] = 0
        remaining_sum = w['weights'].sum()
        difference = 1 - remaining_sum

        non_zero_count = (w['weights'] > 0).sum()
        if non_zero_count > 0:
            w.loc[w['weights'] > 0, 'weights'] += difference / non_zero_count

        # Ensure the sum of adjusted weights is 1
        w['weights'] /= w['weights'].sum()
        # w.sort_values(by="weights", ascending=False)

        return w

    def calculate_cumulative_return(self, allocation, stockData, startDate):
        ret = 0
        ret_hist = []

        for i in range(len(stockData[(stockData.index > startDate)])):
            r = np.sum(stockData[(stockData.index > startDate)][allocation.index].pct_change().iloc[i] * allocation.T,
                       axis=1)
            ret += r.weights
            ret_hist.append(r.weights)

        return ret, np.cumsum(ret_hist)

    # sharpe, downside volatility, max drawdown, avg return
    def calculate_porfolio_metrics(self, cumulative_return):
        returns = pd.Series(cumulative_return).pct_change().dropna()
        returns.replace([np.inf, -np.inf], 0, inplace=True)

        prices = pd.DataFrame(cumulative_return, columns=["prices"])

        # sharpe ratio
        annual_risk_free_rate = 0.02
        # Convert annual risk-free rate to daily (assuming 252 trading days in a year)
        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1
        average_daily_return = returns.mean()
        excess_daily_return = average_daily_return - daily_risk_free_rate
        std_dev_daily = returns.std()
        daily_sharpe_ratio = excess_daily_return / std_dev_daily
        annual_sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)

        # downside volatility
        MAR = 0
        downside_deviation = returns.apply(lambda x: min(x - MAR, 0))
        squared_downside_deviation = downside_deviation ** 2
        mean_squared_downside_deviation = squared_downside_deviation.mean()
        downside_volatility = np.sqrt(mean_squared_downside_deviation)

        # max drawdown
        prices['rolling_max'] = prices['prices'].rolling(20, min_periods=1).max()
        prices['drawdown'] = (prices['prices'] - prices['rolling_max']) / prices['rolling_max']
        prices['max_drawdown'] = prices['drawdown'].rolling(20, min_periods=1).min()
        max_dd = prices.dropna(subset=["max_drawdown"])["max_drawdown"].min()

        # avg return
        filtered_returns = returns[returns < 1]
        avg_return = filtered_returns.mean()

        return annual_sharpe_ratio, downside_volatility, max_dd, avg_return
