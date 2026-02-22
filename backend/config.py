"""
backend/config.py
=================
Central configuration using pydantic-settings.
All env variables are type-safe and validated on startup.
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
import pytz


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Financial APIs ────────────────────────────────────
    alpha_vantage_key: str = ""
    news_api_key: str = ""
    fmp_api_key: str = ""

    # ── Database ──────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/finbot"
    sync_database_url: str = "postgresql://postgres:password@localhost:5432/finbot"
    redis_url: str = "redis://localhost:6379/0"

    # ── App ───────────────────────────────────────────────
    app_env: str = "development"
    secret_key: str = "dev-secret-key-change-in-prod"
    allowed_origins: str = "http://localhost:3000"

    # ── LangSmith ────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "finbot-financial-agent"

    # ── Rate Limits ───────────────────────────────────────
    alpha_vantage_rpm: int = 5
    news_api_rpd: int = 100
    fmp_rpd: int = 250

    # ── Cache TTL ─────────────────────────────────────────
    stock_price_ttl: int = 60
    news_ttl: int = 300
    fundamentals_ttl: int = 3600
    market_data_ttl: int = 300

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    class Config:
        env_file = ".env"
        extra = "ignore"


class IndianMarket:
    """Indian stock market constants."""
    IST = pytz.timezone("Asia/Kolkata")
    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"
    OPEN_HOUR, OPEN_MIN = 9, 15
    CLOSE_HOUR, CLOSE_MIN = 15, 30

    SECTORS = {
        "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "FMCG":    ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
        "Auto":    ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
        "Pharma":  ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS"],
        "Energy":  ["RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS"],
    }

    NIFTY50 = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BAJFINANCE.NS","KOTAKBANK.NS",
        "BHARTIARTL.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
        "TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","SUNPHARMA.NS","NESTLEIND.NS",
    ]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
indian_market = IndianMarket()
