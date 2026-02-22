import pandas as pd
import numpy as np
from finance.portfolio_optimizer import PortfolioOptimizer

def test_optimizer_weights_sum_to_one():
    returns = pd.DataFrame(np.random.randn(100, 3) / 100)
    optimizer = PortfolioOptimizer(returns)
    weights = optimizer.optimize()

    assert round(sum(weights), 5) == 1.0