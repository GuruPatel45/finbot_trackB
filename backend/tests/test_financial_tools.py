import json
from backend.tools.financial_tools import fetch_stock_price


def test_stock_price_returns_data():
    data = fetch_stock_price("AAPL")
    data = json.loads(data)

    assert data is not None
    assert "price" in data
    assert isinstance(data["price"], (int, float))