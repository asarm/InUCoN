'''
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
'''

import numpy as np
import pandas as pd
import riskfolio as rp
from dateutil.relativedelta import relativedelta
from datetime import datetime

class Allocator:
    def __init__(self, rebalancing_period=30, lookback_months=6):
        self.rebalancing_period = rebalancing_period
        self.lookback_months = lookback_months

    def mean_variance_optimization(self, data, min_weight=0.3, take_pct_change=True):
        if take_pct_change:
            data = data.pct_change().dropna()

        port = rp.Portfolio(data)
        method_mu = 'hist'
        method_cov = 'hist'
        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

        model = 'Classic'
        rm = 'MV'
        obj = 'Sharpe'
        hist = True
        rf = 0
        l = 0

        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        w.loc[w['weights'] < min_weight, 'weights'] = 0
        remaining_sum = w['weights'].sum()
        difference = 1 - remaining_sum

        non_zero_count = (w['weights'] > 0).sum()
        if non_zero_count > 0:
            w.loc[w['weights'] > 0, 'weights'] += difference / non_zero_count

        w['weights'] /= w['weights'].sum()
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

    def calculate_porfolio_metrics(self, cumulative_return):
        returns = pd.Series(cumulative_return).pct_change().dropna()
        returns.replace([np.inf, -np.inf], 0, inplace=True)

        prices = pd.DataFrame(cumulative_return, columns=["prices"])

        annual_risk_free_rate = 0.02
        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1
        average_daily_return = returns.mean()
        excess_daily_return = average_daily_return - daily_risk_free_rate
        std_dev_daily = returns.std()
        daily_sharpe_ratio = excess_daily_return / std_dev_daily
        annual_sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)

        MAR = 0
        downside_deviation = returns.apply(lambda x: min(x - MAR, 0))
        squared_downside_deviation = downside_deviation ** 2
        mean_squared_downside_deviation = squared_downside_deviation.mean()
        downside_volatility = np.sqrt(mean_squared_downside_deviation)

        prices['rolling_max'] = prices['prices'].rolling(20, min_periods=1).max()
        prices['drawdown'] = (prices['prices'] - prices['rolling_max']) / prices['rolling_max']
        prices['max_drawdown'] = prices['drawdown'].rolling(20, min_periods=1).min()
        max_dd = prices.dropna(subset=["max_drawdown"])["max_drawdown"].min()

        filtered_returns = returns[returns < 1]
        avg_return = filtered_returns.mean()

        return annual_sharpe_ratio, downside_volatility, max_dd, avg_return

    def _get_lookback_date(self, current_date):
        """Helper function to calculate lookback date in string format"""
        if isinstance(current_date, str):
            dt = datetime.strptime(current_date, '%Y-%m-%d')
        else:
            dt = current_date
        lookback_dt = dt - relativedelta(months=self.lookback_months)
        return lookback_dt.strftime('%Y-%m-%d')

    def calculate_rebalanced_returns(self, stockData, startDate, initial_investment=100000):
        """
        Calculate portfolio returns with periodic rebalancing using 6-month lookback window
        
        Args:
            stockData (pd.DataFrame): Historical price data for all assets
            startDate (str): Start date in 'YYYY-MM-DD' format
            initial_investment (float): Initial investment amount
        """
        portfolio_value = initial_investment
        cumulative_returns = []
        weights_history = []
        current_weights = None
        rebalance_dates = []

        filtered_data = stockData[stockData.index > startDate]
        
        for i in range(len(filtered_data)):
            current_date = filtered_data.index[i]
            current_date_str = current_date if isinstance(current_date, str) else current_date.strftime('%Y-%m-%d')
            
            # Rebalance on day 1 and every rebalancing_period days
            if i == 0 or i % self.rebalancing_period == 0:
                # Calculate the start of the lookback window
                lookback_start = self._get_lookback_date(current_date_str)
                
                # Get data for the lookback window
                historical_data = stockData[
                    (stockData.index <= current_date_str) & 
                    (stockData.index >= lookback_start)
                ]
        
                # Skip rebalancing if we don't have enough historical data
                if len(historical_data) < 20:  # Minimum required days for meaningful optimization
                    if current_weights is None:
                        raise ValueError("Not enough historical data for initial optimization")
                    continue
                
                rebalance_dates.append(current_date_str)
                new_weights = self.mean_variance_optimization(historical_data, min_weight=0.01)
                current_weights = new_weights
                weights_history.append((current_date_str, new_weights))
            
            # Calculate daily returns
            daily_returns = filtered_data.pct_change().iloc[i]
            portfolio_return = np.sum(daily_returns * current_weights['weights'])
            portfolio_value *= (1 + portfolio_return)
            cumulative_returns.append(portfolio_value)
        
        # Convert weights history to DataFrame
        weights_df = pd.DataFrame(
            [w[1]['weights'].to_dict() for w in weights_history],
            index=[w[0] for w in weights_history]
        )
        
        final_return = (portfolio_value - initial_investment) / initial_investment
        cumulative_returns = pd.Series(cumulative_returns, index=filtered_data.index)
        
        return final_return, cumulative_returns, weights_df, rebalance_dates

    def get_latest_weights(self, stockData, current_date=None):
        """
        Get optimal weights using the most recent 6 months of data
        
        Args:
            stockData (pd.DataFrame): Historical price data for all assets
            current_date (str, optional): Date in 'YYYY-MM-DD' format
        """
        if current_date is None:
            last_date = stockData.index[-1]
            current_date = last_date if isinstance(last_date, str) else last_date.strftime('%Y-%m-%d')
            
        lookback_start = self._get_lookback_date(current_date)
        
        historical_data = stockData[
            (stockData.index <= current_date) & 
            (stockData.index >= lookback_start)
        ]

        if len(historical_data) < 20:
            raise ValueError("Not enough historical data for optimization")
            
        return self.mean_variance_optimization(historical_data, min_weight=0.01)