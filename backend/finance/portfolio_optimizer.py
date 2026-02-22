import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        returns: DataFrame of historical returns
        risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()

    def portfolio_performance(self, weights):
        returns = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return returns, volatility

    def negative_sharpe_ratio(self, weights):
        p_return, p_vol = self.portfolio_performance(weights)
        sharpe = (p_return - self.risk_free_rate) / p_vol
        return -sharpe

    def optimize(self):
        num_assets = len(self.mean_returns)
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))

        initial_weights = num_assets * [1. / num_assets]

        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x