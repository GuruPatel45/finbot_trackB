import pytest
from backend.tools.financial_tools import get_stock_price

@pytest.mark.asyncio
async def test_stock_price_returns_data():
    data = await get_stock_price("AAPL")
    assert data is not None
    assert "price" in data
    assert isinstance(data["price"], (int, float))