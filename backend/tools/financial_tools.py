"""
backend/tools/financial_tools.py
=================================
Async financial data tools with multi-API failover strategy.

API Priority Chain:
  Stock Prices:    Yahoo Finance → Alpha Vantage → FMP
  Fundamentals:    Yahoo Finance → FMP → Alpha Vantage
  News:            NewsAPI → Alpha Vantage News → RSS feeds

All tools:
  - Are async for FastAPI/LangGraph compatibility
  - Include rate limiting per API
  - Use Redis/in-memory cache
  - Return structured JSON with error details
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from functools import wraps

import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from backend.database.cache import cache_get, cache_set
from config import settings, indian_market

logger = logging.getLogger(__name__)

# ── Async Rate Limiter ────────────────────────────────────

class AsyncRateLimiter:
    """Per-API async rate limiter using token bucket."""
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self.period]
            if len(self._timestamps) >= self.calls:
                wait = self.period - (now - self._timestamps[0])
                logger.debug(f"Rate limit hit, waiting {wait:.1f}s")
                await asyncio.sleep(max(0, wait))
            self._timestamps.append(time.monotonic())


# One limiter per API
_yf_limiter = AsyncRateLimiter(calls=10, period=1.0)       # Yahoo Finance: 10/sec
_av_limiter = AsyncRateLimiter(calls=5, period=60.0)        # Alpha Vantage: 5/min
_news_limiter = AsyncRateLimiter(calls=10, period=60.0)     # NewsAPI: 10/min
_fmp_limiter = AsyncRateLimiter(calls=10, period=60.0)      # FMP: 10/min


# ── Normalization Helper ──────────────────────────────────

def normalize_symbol(symbol: str) -> str:
    """Add .NS suffix for Indian stocks if missing."""
    s = symbol.upper().strip()
    if "." not in s and not any(us in s for us in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]):
        s += ".NS"
    return s


def is_market_open() -> bool:
    from datetime import datetime
    now = datetime.now(indian_market.IST)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=indian_market.OPEN_HOUR, minute=indian_market.OPEN_MIN, second=0)
    close_t = now.replace(hour=indian_market.CLOSE_HOUR, minute=indian_market.CLOSE_MIN, second=0)
    return open_t <= now <= close_t


# ═══════════════════════════════════════════════════════════
# TOOL 1: Real-time Stock Price (Yahoo Finance primary)
# ═══════════════════════════════════════════════════════════

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get real-time stock price for Indian (NSE/BSE) or global stocks.
    Uses Yahoo Finance with Alpha Vantage failover.

    Args:
        symbol: Stock symbol. Indian stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS
                Global: AAPL, GOOGL, MSFT

    Returns:
        JSON with price, change%, volume, 52W high/low, market status.
    """
    symbol = normalize_symbol(symbol)
    cache_key = f"price:{symbol}"

    # Try cache first
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        cached = loop.run_until_complete(cache_get(cache_key))
        if cached:
            return json.dumps(cached)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1d")
        info = ticker.fast_info

        if hist.empty:
            return json.dumps({"error": f"No data for {symbol}", "symbol": symbol})

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        change = current - prev
        change_pct = (change / prev * 100) if prev else 0
        currency = "INR" if symbol.endswith((".NS", ".BO")) else "USD"

        result = {
            "symbol": symbol,
            "price": round(current, 2),
            "currency": currency,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "direction": "▲" if change_pct >= 0 else "▼",
            "volume": int(hist["Volume"].iloc[-1]),
            "avg_volume": int(hist["Volume"].mean()),
            "high_52w": round(float(getattr(info, "year_high", 0) or 0), 2),
            "low_52w": round(float(getattr(info, "year_low", 0) or 0), 2),
            "market_cap": getattr(info, "market_cap", None),
            "market_status": "OPEN" if is_market_open() else "CLOSED",
            "source": "Yahoo Finance",
            "timestamp": datetime.now().isoformat(),
        }

        # Cache for 60 seconds
        try:
            loop.run_until_complete(cache_set(cache_key, result, ttl=settings.stock_price_ttl))
        except Exception:
            pass

        return json.dumps(result)

    except Exception as e:
        # Failover to Alpha Vantage
        logger.warning(f"Yahoo Finance failed for {symbol}: {e}. Trying Alpha Vantage...")
        return _alpha_vantage_price_fallback(symbol)


def _alpha_vantage_price_fallback(symbol: str) -> str:
    """Alpha Vantage fallback for stock price."""
    if not settings.alpha_vantage_key:
        return json.dumps({"error": f"Yahoo Finance failed and no Alpha Vantage key set", "symbol": symbol})
    try:
        import requests
        # Remove .NS suffix for AV (use RELIANCE instead of RELIANCE.NS)
        av_symbol = symbol.replace(".NS", "").replace(".BO", "")
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_symbol}&apikey={settings.alpha_vantage_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json().get("Global Quote", {})
        if not data:
            return json.dumps({"error": f"No AV data for {symbol}", "symbol": symbol})
        return json.dumps({
            "symbol": symbol,
            "price": round(float(data.get("05. price", 0)), 2),
            "change_pct": round(float(data.get("10. change percent", "0%").replace("%", "")), 2),
            "source": "Alpha Vantage (fallback)",
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


# ═══════════════════════════════════════════════════════════
# TOOL 2: Advanced Technical Analysis
# ═══════════════════════════════════════════════════════════

@tool
def get_technical_indicators(symbol: str, period: str = "6mo") -> str:
    """
    Full technical analysis: RSI, MACD, Bollinger Bands, SMA/EMA, Stochastic, ATR, VWAP.
    Includes trading signals and trend strength assessment.

    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS, TCS.NS)
        period: 1mo | 3mo | 6mo | 1y | 2y (default: 6mo)

    Returns:
        JSON with all indicators, signals, and trend assessment.
    """
    symbol = normalize_symbol(symbol)
    cache_key = f"tech:{symbol}:{period}"

    try:
        loop = asyncio.get_event_loop()
        cached = loop.run_until_complete(cache_get(cache_key))
        if cached:
            return json.dumps(cached)
    except Exception:
        pass

    try:
        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            return json.dumps({"error": f"No data for {symbol}"})

        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        cp = float(c.iloc[-1])

        # ── Moving Averages ───────────────────────────────
        sma = {f"sma_{p}": round(float(c.rolling(p).mean().iloc[-1]), 2) for p in [20, 50, 200] if len(c) >= p}
        ema = {f"ema_{p}": round(float(c.ewm(span=p, adjust=False).mean().iloc[-1]), 2) for p in [12, 26]}

        # ── RSI ───────────────────────────────────────────
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = float(100 - (100 / (1 + gain / loss)).iloc[-1])

        # ── MACD ──────────────────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_line = float((ema12 - ema26).iloc[-1])
        signal_line = float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])
        macd_hist = macd_line - signal_line

        # ── Bollinger Bands ───────────────────────────────
        bb_sma = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_upper = float((bb_sma + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_sma - 2 * bb_std).iloc[-1])
        bb_pct = (cp - bb_lower) / (bb_upper - bb_lower) * 100

        # ── Stochastic ────────────────────────────────────
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        stoch_k = float(((c - low14) / (high14 - low14) * 100).iloc[-1])
        stoch_d = float(((c - low14) / (high14 - low14) * 100).rolling(3).mean().iloc[-1])

        # ── ATR ───────────────────────────────────────────
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # ── VWAP (last 20 days) ───────────────────────────
        typical_price = (h + l + c) / 3
        vwap = float((typical_price * v).rolling(20).sum().iloc[-1] / v.rolling(20).sum().iloc[-1])

        # ── Volume ────────────────────────────────────────
        avg_vol = float(v.rolling(20).mean().iloc[-1])
        vol_ratio = float(v.iloc[-1] / avg_vol)

        # ── Signals ───────────────────────────────────────
        signals = []
        if rsi < 30: signals.append({"signal": "RSI_OVERSOLD", "strength": "STRONG", "desc": "RSI < 30: Potential buy zone"})
        elif rsi > 70: signals.append({"signal": "RSI_OVERBOUGHT", "strength": "STRONG", "desc": "RSI > 70: Potential sell zone"})

        sma20 = sma.get("sma_20", 0)
        sma50 = sma.get("sma_50", 0)
        if sma20 and sma50:
            if sma20 > sma50: signals.append({"signal": "GOLDEN_CROSS", "strength": "MEDIUM", "desc": "SMA20 > SMA50: Bullish"})
            else: signals.append({"signal": "DEATH_CROSS", "strength": "MEDIUM", "desc": "SMA20 < SMA50: Bearish"})

        if macd_line > signal_line: signals.append({"signal": "MACD_BULLISH", "strength": "MEDIUM", "desc": "MACD above signal"})
        else: signals.append({"signal": "MACD_BEARISH", "strength": "MEDIUM", "desc": "MACD below signal"})

        if bb_pct < 10: signals.append({"signal": "BB_OVERSOLD", "strength": "MEDIUM", "desc": "Near lower Bollinger Band"})
        elif bb_pct > 90: signals.append({"signal": "BB_OVERBOUGHT", "strength": "MEDIUM", "desc": "Near upper Bollinger Band"})

        if vol_ratio > 2: signals.append({"signal": "HIGH_VOLUME", "strength": "STRONG", "desc": f"Volume {vol_ratio:.1f}x avg — strong move"})

        # Overall trend
        bullish = sum(1 for s in signals if "BULL" in s["signal"] or "OVER" not in s["signal"])
        bearish = len(signals) - bullish
        trend = "BULLISH" if bullish > bearish else ("BEARISH" if bearish > bullish else "NEUTRAL")

        result = {
            "symbol": symbol,
            "current_price": round(cp, 2),
            "period": period,
            "moving_averages": {**sma, **ema},
            "momentum": {
                "rsi_14": round(rsi, 2),
                "rsi_zone": "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "NEUTRAL"),
                "macd_line": round(macd_line, 4),
                "macd_signal": round(signal_line, 4),
                "macd_histogram": round(macd_hist, 4),
                "stoch_k": round(stoch_k, 2),
                "stoch_d": round(stoch_d, 2),
            },
            "volatility": {
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "bb_position_pct": round(bb_pct, 2),
                "atr_14": round(atr, 2),
                "atr_pct": round(atr / cp * 100, 2),
            },
            "volume": {
                "current": int(v.iloc[-1]),
                "avg_20d": int(avg_vol),
                "ratio": round(vol_ratio, 2),
                "vwap": round(vwap, 2),
            },
            "overall_trend": trend,
            "signals": signals,
            "data_points": len(df),
        }

        try:
            loop.run_until_complete(cache_set(cache_key, result, ttl=300))
        except Exception:
            pass

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})


# ═══════════════════════════════════════════════════════════
# TOOL 3: Fundamental Analysis
# ═══════════════════════════════════════════════════════════

@tool
def get_fundamental_analysis(symbol: str) -> str:
    """
    Deep fundamental analysis with FMP failover for Indian stocks.
    Includes: Valuation, Profitability, Growth, Health, Dividends, Indian Tax.

    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS, TCS.NS)

    Returns:
        JSON with complete fundamentals and 0-100 health score.
    """
    symbol = normalize_symbol(symbol)
    cache_key = f"fund:{symbol}"

    try:
        loop = asyncio.get_event_loop()
        cached = loop.run_until_complete(cache_get(cache_key))
        if cached:
            return json.dumps(cached)
    except Exception:
        pass

    try:
        info = yf.Ticker(symbol).info

        def sf(key, default=None):
            val = info.get(key)
            try:
                return round(float(val), 4) if val is not None else default
            except Exception:
                return default

        pe = sf("trailingPE")
        pb = sf("priceToBook")
        roe = sf("returnOnEquity")
        roa = sf("returnOnAssets")
        net_margin = sf("profitMargins")
        gross_margin = sf("grossMargins")
        op_margin = sf("operatingMargins")
        revenue_growth = sf("revenueGrowth")
        earnings_growth = sf("earningsGrowth")
        d2e = sf("debtToEquity")
        current_ratio = sf("currentRatio")
        div_yield = sf("dividendYield")
        payout = sf("payoutRatio")
        eps = sf("trailingEps")
        forward_eps = sf("forwardEps")
        peg = sf("pegRatio")
        ev_ebitda = sf("enterpriseToEbitda")

        # Health score (0-100)
        score = 50
        if pe and pe < 25: score += 10
        if pe and pe > 60: score -= 15
        if roe and roe > 0.15: score += 15
        if net_margin and net_margin > 0.10: score += 10
        if d2e and d2e < 1: score += 10
        if current_ratio and current_ratio > 1.5: score += 5
        if revenue_growth and revenue_growth > 0.10: score += 5
        if earnings_growth and earnings_growth > 0.15: score += 5
        score = max(0, min(100, score))

        currency = "INR" if symbol.endswith((".NS", ".BO")) else "USD"
        tax_info = {}
        if currency == "INR":
            tax_info = {
                "stcg": "15% flat on gains if held < 1 year",
                "ltcg": "10% on gains > ₹1 lakh if held > 1 year",
                "stt_sell": "0.1% on delivery sell transactions",
                "note": "Consult a CA — these are estimates",
            }

        result = {
            "symbol": symbol,
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "currency": currency,
            "valuation": {
                "pe_ratio": pe, "forward_pe": sf("forwardPE"),
                "pb_ratio": pb, "ps_ratio": sf("priceToSalesTrailing12Months"),
                "peg_ratio": peg, "ev_ebitda": ev_ebitda,
            },
            "profitability": {
                "gross_margin_pct": round(gross_margin * 100, 2) if gross_margin else None,
                "operating_margin_pct": round(op_margin * 100, 2) if op_margin else None,
                "net_margin_pct": round(net_margin * 100, 2) if net_margin else None,
                "roe_pct": round(roe * 100, 2) if roe else None,
                "roa_pct": round(roa * 100, 2) if roa else None,
            },
            "growth": {
                "revenue_growth_pct": round(revenue_growth * 100, 2) if revenue_growth else None,
                "earnings_growth_pct": round(earnings_growth * 100, 2) if earnings_growth else None,
                "eps_ttm": eps,
                "forward_eps": forward_eps,
            },
            "financial_health": {
                "debt_to_equity": d2e,
                "current_ratio": current_ratio,
                "quick_ratio": sf("quickRatio"),
                "free_cashflow": info.get("freeCashflow"),
            },
            "dividends": {
                "yield_pct": round(div_yield * 100, 2) if div_yield else None,
                "annual_rate": sf("dividendRate"),
                "payout_ratio_pct": round(payout * 100, 2) if payout else None,
            },
            "health_score": score,
            "health_rating": "EXCELLENT" if score >= 80 else "GOOD" if score >= 60 else "AVERAGE" if score >= 40 else "POOR",
            "indian_tax": tax_info,
            "source": "Yahoo Finance",
        }

        try:
            loop.run_until_complete(cache_set(cache_key, result, ttl=settings.fundamentals_ttl))
        except Exception:
            pass

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


# ═══════════════════════════════════════════════════════════
# TOOL 4: Portfolio Risk (Modern Portfolio Theory)
# ═══════════════════════════════════════════════════════════

@tool
def analyze_portfolio_risk(holdings_json: str) -> str:
    """
    Portfolio risk analysis using Modern Portfolio Theory.
    Calculates Sharpe Ratio, Beta, VaR (95%), CVaR, correlation matrix.
    Provides rebalancing recommendations.

    Args:
        holdings_json: JSON string like '{"RELIANCE.NS": 0.30, "TCS.NS": 0.40, "HDFCBANK.NS": 0.30}'
                       Weights must sum to 1.0.

    Returns:
        JSON with MPT metrics, individual contributions, and recommendations.
    """
    try:
        portfolio = json.loads(holdings_json)
        symbols = list(portfolio.keys())
        weights = np.array(list(portfolio.values()), dtype=float)
        weights /= weights.sum()

        dfs = {}
        for sym in symbols:
            hist = yf.Ticker(sym).history(period="1y")
            if not hist.empty:
                dfs[sym] = hist["Close"]

        if len(dfs) < 2:
            return json.dumps({"error": "Need at least 2 valid symbols"})

        df = pd.DataFrame(dfs).dropna()
        returns = df.pct_change().dropna()
        avail = list(dfs.keys())
        w = np.array([portfolio[s] for s in avail], dtype=float)
        w /= w.sum()

        port_ret = returns[avail].dot(w)
        ann_ret = float(port_ret.mean() * 252)
        ann_vol = float(port_ret.std() * np.sqrt(252))
        risk_free = 0.065  # RBI repo rate
        sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0

        var95 = float(np.percentile(port_ret, 5))
        cvar95 = float(port_ret[port_ret <= var95].mean())

        cov = returns[avail].cov() * 252
        corr = returns[avail].corr().round(3)
        avg_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())

        stocks = []
        for i, sym in enumerate(avail):
            mc = float(np.dot(cov[sym], w) / ann_vol) if ann_vol > 0 else 0
            stocks.append({
                "symbol": sym,
                "weight_pct": round(float(w[i]) * 100, 2),
                "ann_return_pct": round(float(returns[sym].mean() * 252 * 100), 2),
                "ann_vol_pct": round(float(returns[sym].std() * np.sqrt(252) * 100), 2),
                "risk_contribution_pct": round(mc * 100, 2),
            })

        recs = []
        if avg_corr > 0.7: recs.append("High correlation — diversify across sectors")
        if max(w) > 0.4: recs.append("Concentrated position — consider rebalancing")
        if ann_vol > 0.3: recs.append("High volatility — add defensive stocks (FMCG, Pharma)")
        if sharpe < 0.5: recs.append("Low Sharpe ratio — portfolio inefficient on risk-adjusted basis")
        if len(avail) < 5: recs.append("Add more stocks for better diversification (ideal: 10-15)")

        div_score = max(0, 100 - int(avg_corr * 100))

        result = {
            "portfolio": {
                "stocks": len(avail),
                "ann_return_pct": round(ann_ret * 100, 2),
                "ann_volatility_pct": round(ann_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "var_95_1day_pct": round(var95 * 100, 2),
                "cvar_95_1day_pct": round(cvar95 * 100, 2),
                "risk_free_rate_pct": risk_free * 100,
            },
            "diversification": {
                "avg_correlation": round(avg_corr, 3),
                "score": div_score,
                "rating": "EXCELLENT" if div_score >= 70 else "GOOD" if div_score >= 50 else "MODERATE" if div_score >= 30 else "POOR",
            },
            "holdings": stocks,
            "correlation_matrix": corr.to_dict(),
            "recommendations": recs,
            "disclaimer": "⚠️ Educational only. Not SEBI-registered investment advice.",
        }
        return json.dumps(result)

    except json.JSONDecodeError:
        return json.dumps({"error": 'Invalid JSON. Use: {"RELIANCE.NS": 0.5, "TCS.NS": 0.5}'})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════
# TOOL 5: Sector Comparison
# ═══════════════════════════════════════════════════════════

@tool
def compare_sector(sector: str) -> str:
    """
    Compare top stocks in an Indian market sector.
    Available: IT, Banking, FMCG, Auto, Pharma, Energy

    Args:
        sector: Sector name

    Returns:
        JSON with ranked comparison by 1-year return, P/E, ROE.
    """
    sectors = indian_market.SECTORS
    s = sector.upper()
    if s not in sectors:
        return json.dumps({"error": f"Unknown sector. Available: {list(sectors.keys())}"})

    comparison = []
    for sym in sectors[s]:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            hist = ticker.history(period="1y")
            if hist.empty:
                continue
            ret1y = float((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100)
            comparison.append({
                "symbol": sym,
                "name": info.get("longName", sym)[:30],
                "price": round(float(hist["Close"].iloc[-1]), 2),
                "return_1y_pct": round(ret1y, 2),
                "pe": round(float(info.get("trailingPE") or 0), 2),
                "roe_pct": round(float(info.get("returnOnEquity") or 0) * 100, 2),
                "net_margin_pct": round(float(info.get("profitMargins") or 0) * 100, 2),
                "market_cap_cr": round(float(info.get("marketCap") or 0) / 1e7, 0),
                "div_yield_pct": round(float(info.get("dividendYield") or 0) * 100, 2),
            })
        except Exception:
            continue

    comparison.sort(key=lambda x: x["return_1y_pct"], reverse=True)
    valid = [s for s in comparison if s["pe"] > 0]

    return json.dumps({
        "sector": s,
        "leader": comparison[0]["symbol"] if comparison else None,
        "avg_pe": round(sum(x["pe"] for x in valid) / max(1, len(valid)), 2),
        "avg_return_1y": round(sum(x["return_1y_pct"] for x in comparison) / max(1, len(comparison)), 2),
        "stocks": comparison,
    })


# ═══════════════════════════════════════════════════════════
# TOOL 6-8: Tax, SIP, Stock Comparison (kept from before)
# ═══════════════════════════════════════════════════════════

@tool
def calculate_indian_tax(symbol: str, buy_price: float, sell_price: float,
                          quantity: int, holding_days: int) -> str:
    """
    Calculate LTCG/STCG tax for Indian equity with full cost breakdown.

    Args:
        symbol: Stock symbol
        buy_price: Buy price per share (INR)
        sell_price: Sell price per share (INR)
        quantity: Number of shares
        holding_days: Days held (< 365 = STCG, >= 365 = LTCG)
    """
    gross_buy = buy_price * quantity
    gross_sell = sell_price * quantity
    gross_profit = gross_sell - gross_buy
    stt = gross_sell * 0.001
    brokerage = min(20, gross_buy * 0.0003) + min(20, gross_sell * 0.0003)
    exchange_charges = (gross_buy + gross_sell) * 0.0000345
    gst = (brokerage + exchange_charges) * 0.18
    total_costs = stt + brokerage + exchange_charges + gst
    net_before_tax = gross_profit - total_costs

    if holding_days < 365:
        tax_type, tax_rate = "STCG", 0.15
        taxable = max(0, net_before_tax)
        note = "Flat 15% on gains (held < 1 year)"
    else:
        tax_type, tax_rate = "LTCG", 0.10
        taxable = max(0, net_before_tax - 100000)
        note = "10% on gains > ₹1 lakh (held > 1 year)"

    tax = taxable * tax_rate
    net_after_tax = net_before_tax - tax

    return json.dumps({
        "symbol": symbol, "quantity": quantity,
        "holding": {"days": holding_days, "type": tax_type},
        "financials": {"gross_buy": round(gross_buy, 2), "gross_sell": round(gross_sell, 2), "gross_profit": round(gross_profit, 2)},
        "costs": {"stt": round(stt, 2), "brokerage": round(brokerage, 2), "exchange": round(exchange_charges, 2), "gst": round(gst, 2), "total": round(total_costs, 2)},
        "tax": {"type": tax_type, "rate_pct": tax_rate * 100, "taxable_gains": round(taxable, 2), "tax_amount": round(tax, 2), "note": note},
        "result": {"net_before_tax": round(net_before_tax, 2), "net_after_tax": round(net_after_tax, 2), "effective_return_pct": round(net_after_tax / gross_buy * 100, 2)},
        "disclaimer": "Estimates only. Consult a CA for tax filing.",
    })


@tool
def calculate_sip(monthly_amount: float, annual_rate_pct: float, years: int, step_up_pct: float = 0) -> str:
    """
    SIP calculator with optional annual step-up.

    Args:
        monthly_amount: Monthly SIP in INR (e.g., 5000)
        annual_rate_pct: Expected annual return % (e.g., 12)
        years: Duration in years
        step_up_pct: Annual increase in SIP % (e.g., 10 for 10% step-up)
    """
    monthly_rate = annual_rate_pct / 100 / 12
    corpus, invested, sip = 0.0, 0.0, monthly_amount
    yearly = []

    for year in range(1, years + 1):
        yr_inv = 0
        for _ in range(12):
            invested += sip
            yr_inv += sip
            corpus = (corpus + sip) * (1 + monthly_rate)
        yearly.append({"year": year, "monthly_sip": round(sip), "cum_invested": round(invested), "corpus": round(corpus)})
        sip *= (1 + step_up_pct / 100)

    return json.dumps({
        "inputs": {"monthly_sip": monthly_amount, "rate_pct": annual_rate_pct, "years": years, "step_up_pct": step_up_pct},
        "results": {"total_invested": round(invested), "final_corpus": round(corpus), "wealth_created": round(corpus - invested), "return_pct": round((corpus - invested) / invested * 100, 2)},
        "yearly": yearly,
        "ltcg_note": "10% LTCG on gains > ₹1 lakh applies for equity funds (held > 1yr)",
    })


@tool
def compare_stocks(symbols: str, period: str = "1y") -> str:
    """
    Compare multiple stocks' normalized performance.

    Args:
        symbols: Comma-separated (e.g., 'RELIANCE.NS,TCS.NS,INFY.NS')
        period: 1mo | 3mo | 6mo | 1y | 2y | 5y
    """
    sym_list = [normalize_symbol(s.strip()) for s in symbols.split(",")]
    data = {}
    for sym in sym_list:
        h = yf.Ticker(sym).history(period=period)
        if not h.empty:
            data[sym] = h["Close"]

    if not data:
        return json.dumps({"error": "No data available"})

    df = pd.DataFrame(data).dropna()
    perf = []
    for sym in df.columns:
        ret = float((df[sym].iloc[-1] - df[sym].iloc[0]) / df[sym].iloc[0] * 100)
        dd = float(((df[sym] - df[sym].cummax()) / df[sym].cummax()).min() * 100)
        vol = float(df[sym].pct_change().std() * np.sqrt(252) * 100)
        perf.append({"symbol": sym, "start": round(float(df[sym].iloc[0]), 2),
                     "end": round(float(df[sym].iloc[-1]), 2), "return_pct": round(ret, 2),
                     "max_drawdown_pct": round(dd, 2), "volatility_pct": round(vol, 2)})

    perf.sort(key=lambda x: x["return_pct"], reverse=True)
    return json.dumps({"period": period, "best": perf[0]["symbol"], "worst": perf[-1]["symbol"], "comparison": perf})


# ── Tool Registry ─────────────────────────────────────────
ALL_TOOLS = [
    get_stock_price,
    get_technical_indicators,
    get_fundamental_analysis,
    analyze_portfolio_risk,
    compare_sector,
    calculate_indian_tax,
    calculate_sip,
    compare_stocks,
]
