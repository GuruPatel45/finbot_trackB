import pytest
from backend.tools.financial_tools import fetch_stock_price


async def test_stock_price_returns_data():
    data = await fetch_stock_price("AAPL")
    assert data is not None
    assert "price" in data
    assert isinstance(data["price"], (int, float))