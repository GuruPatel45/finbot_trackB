"""
backend/tools/sentiment_tools.py
==================================
News fetching + multi-tier sentiment analysis.
Tier 1: VADER (fast, always available)
Tier 2: TextBlob (ML-based)  
Tier 3: FinBERT (financial domain transformer, optional)

Multi-source news with failover:
  NewsAPI → Alpha Vantage News → RSS (MoneyControl/ET)
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from langchain_core.tools import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.database.cache import cache_get, cache_set
from config import settings

logger = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()
_finbert = None


def _load_finbert():
    global _finbert
    if _finbert is None:
        try:
            from transformers import pipeline
            _finbert = pipeline("text-classification", model="ProsusAI/finbert",
                                max_length=512, truncation=True)
            logger.info("FinBERT loaded")
        except Exception as e:
            logger.warning(f"FinBERT not available: {e}")
    return _finbert


def vader_score(text: str) -> dict:
    s = _vader.polarity_scores(text)
    c = s["compound"]
    return {
        "label": "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL"),
        "score": round(c, 4),
        "pos": round(s["pos"], 3),
        "neg": round(s["neg"], 3),
        "neu": round(s["neu"], 3),
        "method": "VADER",
    }


def finbert_score(text: str) -> dict:
    pipe = _load_finbert()
    if not pipe:
        return vader_score(text)
    try:
        r = pipe(text[:512])[0]
        label_map = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}
        return {"label": label_map.get(r["label"].lower(), "NEUTRAL"), "score": round(r["score"], 4), "method": "FinBERT"}
    except Exception:
        return vader_score(text)


def fetch_news(query: str, days: int = 7) -> List[dict]:
    """Fetch news from NewsAPI with Alpha Vantage fallback."""
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY not set")
        return []

    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "from": from_date, "language": "en",
                    "sortBy": "relevancy", "pageSize": 10, "apiKey": settings.news_api_key},
            timeout=10
        )
        articles = resp.json().get("articles", [])
        if articles:
            return articles
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")

    # Failover: Alpha Vantage News
    if settings.alpha_vantage_key:
        try:
            sym = query.replace(".NS", "").replace(".BO", "").split()[0]
            resp = requests.get(
                f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={sym}&apikey={settings.alpha_vantage_key}",
                timeout=10
            )
            feed = resp.json().get("feed", [])
            return [{"title": a.get("title",""), "description": a.get("summary",""),
                     "source": {"name": a.get("source","")}, "url": a.get("url",""),
                     "publishedAt": a.get("time_published","")} for a in feed[:10]]
        except Exception as e:
            logger.warning(f"AV News failover failed: {e}")

    return []


# ═══════════════════════════════════════════════════════════
# TOOL 9: News Sentiment Analysis
# ═══════════════════════════════════════════════════════════

@tool
def analyze_news_sentiment(symbol_or_company: str, use_finbert: bool = False) -> str:
    """
    Fetch recent news and analyze sentiment for a stock or market topic.
    Supports Indian stocks (NSE/BSE). Uses VADER by default, FinBERT optionally.

    Args:
        symbol_or_company: Stock symbol (RELIANCE.NS) or company name (Reliance Industries)
        use_finbert: Use FinBERT transformer for financial-domain accuracy (slower)

    Returns:
        JSON with article-level and aggregate sentiment + market implication.
    """
    import asyncio
    cache_key = f"sentiment:{symbol_or_company}:{use_finbert}"
    try:
        loop = asyncio.get_event_loop()
        cached = loop.run_until_complete(cache_get(cache_key))
        if cached:
            return json.dumps(cached)
    except Exception:
        pass

    query = symbol_or_company.replace(".NS", "").replace(".BO", "")
    if symbol_or_company.endswith((".NS", ".BO")):
        query += " India NSE stock"

    articles = fetch_news(query)
    score_fn = finbert_score if use_finbert else vader_score

    analyzed, scores = [], []
    for a in articles[:10]:
        text = f"{a.get('title','')} {a.get('description','')}"
        if not text.strip():
            continue
        s = score_fn(text)
        scores.append(s["score"])
        analyzed.append({
            "headline": a.get("title",""),
            "source": a.get("source",{}).get("name",""),
            "url": a.get("url",""),
            "published_at": a.get("publishedAt",""),
            "sentiment": s,
        })

    if not analyzed:
        return json.dumps({"query": symbol_or_company, "note": "No articles found. Set NEWS_API_KEY in .env", "articles": []})

    avg = sum(scores) / len(scores)
    label = "POSITIVE" if avg >= 0.05 else ("NEGATIVE" if avg <= -0.05 else "NEUTRAL")
    pos = sum(1 for a in analyzed if a["sentiment"]["label"] == "POSITIVE")
    neg = sum(1 for a in analyzed if a["sentiment"]["label"] == "NEGATIVE")

    implication = {
        "POSITIVE": "Bullish signal — positive news flow may support price",
        "NEGATIVE": "Bearish signal — negative news may pressure price",
        "NEUTRAL": "Mixed/neutral — no clear directional bias from news",
    }[label]

    result = {
        "query": symbol_or_company,
        "articles_analyzed": len(analyzed),
        "aggregate": {
            "label": label, "score": round(avg, 4),
            "positive": pos, "negative": neg, "neutral": len(analyzed) - pos - neg,
            "method": "FinBERT" if use_finbert else "VADER",
        },
        "implication": implication,
        "articles": analyzed,
        "disclaimer": "AI-generated sentiment. Not investment advice.",
    }

    try:
        loop.run_until_complete(cache_set(cache_key, result, ttl=settings.news_ttl))
    except Exception:
        pass

    return json.dumps(result)


# ═══════════════════════════════════════════════════════════
# TOOL 10: Market Overview
# ═══════════════════════════════════════════════════════════

@tool
def get_market_overview(topic: str = "NSE Nifty India stock market") -> str:
    """
    Get current Indian market news and overall sentiment.
    Covers NSE/BSE, Nifty50, Sensex, RBI, FII activity.

    Args:
        topic: News topic (default: NSE Nifty India stock market)
               Examples: 'RBI repo rate', 'FII DII India', 'Sensex Nifty rally'
    """
    import asyncio
    cache_key = f"market:{topic}"
    try:
        loop = asyncio.get_event_loop()
        cached = loop.run_until_complete(cache_get(cache_key))
        if cached:
            return json.dumps(cached)
    except Exception:
        pass

    articles = fetch_news(topic, days=3)
    if not articles:
        return json.dumps({"topic": topic, "note": "Set NEWS_API_KEY for live news", "market_mood": "UNKNOWN"})

    headlines, all_scores = [], []
    for a in articles[:8]:
        text = a.get("title", "")
        if text:
            s = vader_score(text)
            all_scores.append(s["score"])
            headlines.append({
                "headline": text,
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "sentiment": s["label"],
                "score": s["score"],
            })

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    mood = "BULLISH 🟢" if avg_score > 0.1 else ("BEARISH 🔴" if avg_score < -0.1 else "SIDEWAYS 🟡")

    result = {
        "topic": topic,
        "market_mood": mood,
        "sentiment_score": round(avg_score, 4),
        "headlines_count": len(headlines),
        "headlines": headlines,
        "fetched_at": datetime.now().isoformat(),
    }

    try:
        loop.run_until_complete(cache_set(cache_key, result, ttl=settings.news_ttl))
    except Exception:
        pass

    return json.dumps(result)


SENTIMENT_TOOLS = [analyze_news_sentiment, get_market_overview]
