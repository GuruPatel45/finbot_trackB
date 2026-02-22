# 📈 FinBot — AI Financial Research Platform
### Track B: Full Industry-Grade Fintech Stack

[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Node-blue)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-orange)](https://console.groq.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%2B%20asyncio-green)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014%20%2B%20TypeScript-black)](https://nextjs.org)

---

## 🏗️ Architecture (Track B Compliant)

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRACK B STACK                              │
├──────────────┬──────────────┬────────────────┬─────────────────┤
│  Frontend    │  Backend     │  Database      │  AI/ML          │
│  Next.js 14  │  FastAPI     │  PostgreSQL    │  LangGraph      │
│  TypeScript  │  asyncio     │  Redis Cache   │  Groq LLM       │
│  Tailwind    │  WebSocket   │  SQLAlchemy 2  │  FinBERT NLP    │
│  Chart.js/D3 │  SSE Stream  │  Alembic       │  VADER          │
├──────────────┴──────────────┴────────────────┴─────────────────┤
│  Deployment: Vercel (FE) + Railway (BE+DB+Redis)               │
│  Monitoring: LangSmith + Custom Dashboard                       │
│  APIs: Yahoo Finance → Alpha Vantage → FMP (failover chain)    │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 LangGraph Multi-Node Workflow

```
START
  │
  ▼
[classify_query] ── LLM classifies query type
  │
  ├── stock_analysis  ──► Price + RSI + MACD + BB + Fundamentals + Sentiment
  ├── portfolio       ──► MPT: Sharpe Ratio + VaR + CVaR + Correlation Matrix  
  ├── news_sentiment  ──► NewsAPI + VADER + FinBERT Transformer
  ├── calculation     ──► SIP Calculator + LTCG/STCG Tax
  ├── sector          ──► IT/Banking/FMCG/Auto/Pharma/Energy Compare
  └── general         ──► Financial concepts, market education
                │
               END
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 16 (or use Docker)
- Redis 7 (or use Docker)

### Option A: Docker (Easiest)

```bash
git clone https://github.com/YOUR_USERNAME/finbot-trackb.git
cd finbot-trackb

# Copy and fill env files
cp backend/.env.example backend/.env
# Add GROQ_API_KEY (free: https://console.groq.com)

docker-compose up
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### Option B: Manual Setup

```bash
# ── Backend ─────────────────────────────────────
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Add your GROQ_API_KEY
uvicorn api.main:app --reload --port 8000

# ── Frontend (new terminal) ──────────────────────
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

---

## 🔑 API Keys Setup

| Key | Source | Free? | Required |
|-----|--------|-------|----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ Yes | ✅ Yes |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org/register) | ✅ 100/day | ⚠️ Recommended |
| `ALPHA_VANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/) | ✅ 25/day | ⚠️ Fallback |
| `FMP_API_KEY` | [financialmodelingprep.com](https://financialmodelingprep.com) | ✅ 250/day | ⚠️ Optional |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | ✅ Free tier | ⚠️ Monitoring |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Main chat (LangGraph) |
| GET | `/api/chat/stream` | SSE streaming response |
| GET | `/api/stocks/{symbol}` | Stock price + technicals |
| GET | `/api/stocks/{symbol}/fundamentals` | Fundamental analysis |
| GET | `/api/sectors/{sector}` | Sector comparison |
| GET | `/api/portfolio/{id}` | Portfolio with live P&L |
| POST | `/api/portfolio/{id}/holdings` | Add holding |
| GET | `/api/alerts` | Price alerts |
| POST | `/api/alerts` | Create alert |
| WS | `/ws/prices` | Real-time price WebSocket |
| GET | `/api/monitoring/stats` | Agent monitoring data |
| GET | `/api/health` | Health check |

---

## 🇮🇳 Indian Market Support

```python
# NSE stocks: .NS suffix
RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS

# BSE stocks: .BO suffix  
RELIANCE.BO, TCS.BO

# Market hours: 9:15 AM – 3:30 PM IST, Mon–Fri
# Tax: STCG 15% (< 1yr) | LTCG 10% on gains > ₹1L (> 1yr)
```

---

## 🚢 Deployment

### Backend → Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

cd backend
railway init
railway up

# Add environment variables in Railway dashboard:
# GROQ_API_KEY, NEWS_API_KEY, LANGCHAIN_API_KEY
# DATABASE_URL and REDIS_URL are auto-set by Railway addons
```

### Frontend → Vercel

```bash
npm install -g vercel
cd frontend
vercel

# Set env variables in Vercel dashboard:
# NEXT_PUBLIC_API_URL = https://your-backend.railway.app
# NEXT_PUBLIC_WS_URL  = wss://your-backend.railway.app
```

---

## 🧪 Testing

```bash
cd backend
pytest tests/ -v --cov=. --cov-report=html
```

---

## 📊 Track B Checklist

- [x] **Next.js + TypeScript + Tailwind** frontend
- [x] **LangGraph** multi-node agent workflow (classify → route → analyze → synthesize)
- [x] **FastAPI** with async/await throughout
- [x] **WebSocket** for real-time price streaming
- [x] **SSE** for streaming agent responses to frontend
- [x] **PostgreSQL** with async SQLAlchemy 2.0
- [x] **Redis** caching with in-memory fallback
- [x] **Multi-API failover** (Yahoo → Alpha Vantage → FMP)
- [x] **Groq LLM** (Llama 3.3 70B) with OpenAI/Anthropic fallback
- [x] **LangSmith** monitoring integration
- [x] **10 financial tools** (price, technicals, fundamentals, portfolio risk, sector, tax, SIP, sentiment, news)
- [x] **FinBERT** transformer for financial sentiment
- [x] **Modern Portfolio Theory** (Sharpe, VaR, CVaR, correlation)
- [x] **Indian market** (NSE/BSE, LTCG/STCG, INR)
- [x] **Docker Compose** for local development
- [x] **Vercel + Railway** deployment configs
- [x] **SEBI compliance** disclaimers throughout

---

*⚠️ Educational analysis only. Not SEBI-registered investment advice.*  
*Powered by: LangGraph · Groq · FastAPI · Next.js · PostgreSQL · Redis*
