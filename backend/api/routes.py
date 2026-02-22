from fastapi import APIRouter
import pandas as pd
import numpy as np
from finance.portfolio_optimizer import PortfolioOptimizer

router = APIRouter()

@router.post("/optimize-portfolio")
async def optimize_portfolio():
    # Dummy data (later we connect real stock returns)
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(252, 4) / 100)

    optimizer = PortfolioOptimizer(returns)
    weights = optimizer.optimize()

    return {
        "optimal_weights": weights.tolist()
    }