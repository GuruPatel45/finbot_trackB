"""
backend/database/models.py
===========================
Async SQLAlchemy 2.0 models for PostgreSQL.
Falls back to SQLite for local dev if Postgres not available.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Float, Boolean, DateTime, JSON, Text, ForeignKey, Integer, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker, AsyncSession
import asyncio
from sqlalchemy.exc import OperationalError

class Base(AsyncAttrs, DeclarativeBase):
    pass


# ── Portfolio & Holdings ──────────────────────────────────

class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), default="My Portfolio")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    holdings: Mapped[List["Holding"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    transactions: Mapped[List["Transaction"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (Index("ix_holdings_portfolio_symbol", "portfolio_id", "symbol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    symbol: Mapped[str] = mapped_column(String(20))        # RELIANCE.NS
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")
    quantity: Mapped[float] = mapped_column(Float)
    avg_buy_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="holdings")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    symbol: Mapped[str] = mapped_column(String(20))
    transaction_type: Mapped[str] = mapped_column(String(4))   # BUY | SELL
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    total_amount: Mapped[float] = mapped_column(Float)
    brokerage: Mapped[float] = mapped_column(Float, default=0.0)
    stt: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="transactions")


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), default="My Watchlist")
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20))
    alert_type: Mapped[str] = mapped_column(String(30))    # PRICE_ABOVE | PRICE_BELOW | RSI_*
    threshold: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ResearchQuery(Base):
    __tablename__ = "research_queries"

    id: Mapped[int] = mapped_column(primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    symbols_mentioned: Mapped[list] = mapped_column(JSON, default=list)
    agent_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tools_used: Mapped[list] = mapped_column(JSON, default=list)
    node_path: Mapped[list] = mapped_column(JSON, default=list)   # LangGraph nodes traversed
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── DB Connection Factory ─────────────────────────────────

def create_db_engine(database_url: str):
    """Create async engine. Auto-detects SQLite vs PostgreSQL."""
    if "sqlite" in database_url:
        # Convert to async sqlite URL
        url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        return create_async_engine(url, echo=False)
    return create_async_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )


async def init_db(engine):
    for i in range(10):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("DB Connected")
            return
        except OperationalError:
            print("Waiting for DB...")
            await asyncio.sleep(2)

    raise Exception("Database not available")


def get_session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
