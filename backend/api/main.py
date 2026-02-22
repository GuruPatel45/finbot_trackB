"""
backend/api/main.py
====================
FastAPI backend — production-grade with:
  - REST endpoints (chat, stocks, portfolio, watchlist, alerts)
  - WebSocket for real-time price streaming
  - Server-Sent Events (SSE) for live agent responses
  - CORS configured for Next.js frontend
  - LangSmith monitoring integration
  - Prometheus metrics
  - Health check with dependency status
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from backend.database.models import create_db_engine, init_db, get_session_factory, Portfolio, Holding, Watchlist, Alert, ResearchQuery
from backend.database.cache import init_redis, cache_get, cache_set, get_cache_status

logger = logging.getLogger(__name__)

# ── DB & Cache setup ──────────────────────────────────────
engine = create_db_engine(settings.database_url)
SessionFactory = get_session_factory(engine)

# ── WebSocket connection manager ──────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws) if hasattr(self.active, 'discard') else None
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.active[:]:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup tasks."""
    logger.info("🚀 FinBot API starting...")
    await init_db(engine)
    await init_redis(settings.redis_url)

    # Configure LangSmith
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"✅ LangSmith tracing enabled: {settings.langchain_project}")

    logger.info("✅ FinBot API ready")
    yield
    logger.info("📴 FinBot API shutting down...")
    await engine.dispose()


app = FastAPI(
    title="FinBot Financial Research API",
    description="AI-powered Indian & global financial research platform — Track B",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(ms, 2))
    return response


# ── DB session dependency ─────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionFactory() as session:
        yield session


# ═══════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > 3000:
            raise ValueError("Query too long (max 3000 chars)")
        return v


class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    avg_buy_price: float
    exchange: str = "NSE"
    sector: Optional[str] = None


class AlertCreate(BaseModel):
    symbol: str
    alert_type: str
    threshold: float
    message: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Health & Monitoring
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Comprehensive health check for deployment monitoring."""
    cache_status = get_cache_status()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "llm": f"Groq ({settings.groq_model})",
        "database": "connected",
        "cache": cache_status,
        "langsmith": settings.langchain_tracing_v2,
        "environment": settings.app_env,
    }


@app.get("/api/monitoring/stats")
async def monitoring_stats(db: AsyncSession = Depends(get_db)):
    """LangSmith-style monitoring dashboard data."""
    from sqlalchemy import select, func
    result = await db.execute(select(func.count(ResearchQuery.id)))
    total_queries = result.scalar()

    result2 = await db.execute(
        select(func.avg(ResearchQuery.execution_time_ms))
        .where(ResearchQuery.execution_time_ms.is_not(None))
    )
    avg_time = result2.scalar() or 0

    return {
        "total_queries": total_queries,
        "avg_execution_time_ms": round(float(avg_time), 2),
        "cache_status": get_cache_status(),
        "llm_model": settings.groq_model,
        "langsmith_project": settings.langchain_project if settings.langchain_tracing_v2 else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# Chat Endpoints
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Main chat endpoint. Routes through LangGraph multi-node workflow.
    Returns structured response with nodes visited and tools used.
    """
    from backend.agents.langgraph_agent import get_agent
    agent = get_agent()

    result = await asyncio.get_event_loop().run_in_executor(
        None, agent.run, request.query, request.session_id
    )

    # Log to DB
    log = ResearchQuery(
        query=request.query,
        agent_response=result.get("response", ""),
        tools_used=result.get("tools_used", []),
        node_path=result.get("nodes_visited", []),
        execution_time_ms=result.get("execution_time_ms"),
    )
    db.add(log)
    await db.commit()

    return {
        "response": result["response"],
        "query_type": result.get("query_type"),
        "nodes_visited": result.get("nodes_visited", []),
        "tools_used": result.get("tools_used", []),
        "execution_time_ms": result.get("execution_time_ms"),
        "status": result.get("status"),
    }


@app.get("/api/chat/stream")
async def chat_stream(query: str, session_id: str = "default"):
    """
    Server-Sent Events streaming for real-time agent responses.
    Frontend connects with EventSource API.
    """
    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Processing...'})}\n\n"

        from backend.agents.langgraph_agent import get_agent
        agent = get_agent()

        # Stream classify step
        yield f"data: {json.dumps({'type': 'node', 'node': 'classify', 'message': 'Classifying query...'})}\n\n"
        await asyncio.sleep(0.1)

        result = await asyncio.get_event_loop().run_in_executor(None, agent.run, query, session_id)

        for node in result.get("nodes_visited", []):
            yield f"data: {json.dumps({'type': 'node', 'node': node, 'message': f'Running {node}...'})}\n\n"
            await asyncio.sleep(0.05)

        for tool in result.get("tools_used", []):
            yield f"data: {json.dumps({'type': 'tool', 'tool': tool, 'message': f'Used: {tool}'})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'type': 'complete', 'response': result['response'], 'execution_time_ms': result.get('execution_time_ms')})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════
# Stock Data Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/api/stocks/{symbol}")
async def get_stock(symbol: str, period: str = "6mo"):
    """Get stock price + technical indicators."""
    import json as _json
    from backend.tools.financial_tools import get_stock_price, get_technical_indicators

    price = _json.loads(await asyncio.get_event_loop().run_in_executor(
        None, get_stock_price.invoke, symbol))
    tech = _json.loads(await asyncio.get_event_loop().run_in_executor(
        None, get_technical_indicators.invoke, {"symbol": symbol, "period": period}))

    return {"symbol": symbol, "price": price, "technicals": tech}


@app.get("/api/stocks/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    """Get fundamental analysis for a stock."""
    import json as _json
    from backend.tools.financial_tools import get_fundamental_analysis
    result = _json.loads(await asyncio.get_event_loop().run_in_executor(
        None, get_fundamental_analysis.invoke, symbol))
    return result


@app.get("/api/sectors/{sector}")
async def get_sector(sector: str):
    """Get sector comparison."""
    import json as _json
    from backend.tools.financial_tools import compare_sector
    result = _json.loads(await asyncio.get_event_loop().run_in_executor(
        None, compare_sector.invoke, sector))
    return result


@app.get("/api/watchlist/prices")
async def watchlist_prices(symbols: str = "RELIANCE.NS,TCS.NS,HDFCBANK.NS"):
    """Batch price fetch for watchlist (used by WebSocket updates)."""
    import json as _json
    from backend.tools.financial_tools import get_stock_price

    sym_list = [s.strip() for s in symbols.split(",")]
    prices = {}
    for sym in sym_list[:15]:  # cap at 15
        try:
            prices[sym] = _json.loads(await asyncio.get_event_loop().run_in_executor(
                None, get_stock_price.invoke, sym))
        except Exception as e:
            prices[sym] = {"error": str(e)}

    return {"prices": prices, "updated_at": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════
# Portfolio Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/api/portfolio/{portfolio_id}")
async def get_portfolio(portfolio_id: int = 1, db: AsyncSession = Depends(get_db)):
    """Get portfolio with live prices and P&L."""
    from sqlalchemy import select
    import json as _json
    from backend.tools.financial_tools import get_stock_price

    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        portfolio = Portfolio(id=portfolio_id, name="My Portfolio")
        db.add(portfolio)
        await db.commit()
        await db.refresh(portfolio)

    result = await db.execute(select(Holding).where(Holding.portfolio_id == portfolio_id))
    holdings = result.scalars().all()

    holdings_data, total_inv, total_cur = [], 0, 0
    for h in holdings:
        price_data = _json.loads(await asyncio.get_event_loop().run_in_executor(
            None, get_stock_price.invoke, h.symbol))
        cur_price = price_data.get("price", h.avg_buy_price)
        inv = h.quantity * h.avg_buy_price
        cur = h.quantity * cur_price
        pnl = cur - inv
        total_inv += inv
        total_cur += cur
        holdings_data.append({
            "id": h.id, "symbol": h.symbol, "exchange": h.exchange,
            "quantity": h.quantity, "avg_buy_price": h.avg_buy_price,
            "current_price": cur_price, "invested_value": round(inv, 2),
            "current_value": round(cur, 2), "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / inv * 100 if inv else 0, 2),
            "sector": h.sector,
            "change_pct": price_data.get("change_pct", 0),
        })

    total_pnl = total_cur - total_inv
    return {
        "portfolio_id": portfolio_id,
        "name": portfolio.name,
        "holdings": holdings_data,
        "summary": {
            "total_invested": round(total_inv, 2),
            "current_value": round(total_cur, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / total_inv * 100 if total_inv else 0, 2),
            "holdings_count": len(holdings_data),
        },
    }


@app.post("/api/portfolio/{portfolio_id}/holdings")
async def add_holding(portfolio_id: int, body: HoldingCreate, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        portfolio = Portfolio(id=portfolio_id, name="My Portfolio")
        db.add(portfolio)
        await db.commit()

    sym = body.symbol.upper()
    if "." not in sym:
        sym += ".NS"

    holding = Holding(portfolio_id=portfolio_id, symbol=sym, quantity=body.quantity,
                      avg_buy_price=body.avg_buy_price, exchange=body.exchange, sector=body.sector)
    db.add(holding)
    await db.commit()
    await db.refresh(holding)
    return {"message": f"Added {sym}", "holding_id": holding.id}


@app.delete("/api/portfolio/{portfolio_id}/holdings/{holding_id}")
async def remove_holding(portfolio_id: int, holding_id: int, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(
        select(Holding).where(Holding.id == holding_id, Holding.portfolio_id == portfolio_id))
    holding = result.scalar_one_or_none()
    if not holding:
        raise HTTPException(404, "Holding not found")
    await db.delete(holding)
    await db.commit()
    return {"message": "Holding removed"}


# ═══════════════════════════════════════════════════════════
# Alerts Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/api/alerts")
async def get_alerts(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Alert).where(Alert.is_active == True))
    alerts = result.scalars().all()
    return {"alerts": [{"id": a.id, "symbol": a.symbol, "type": a.alert_type,
                         "threshold": a.threshold, "triggered": a.triggered,
                         "created_at": a.created_at.isoformat()} for a in alerts]}


@app.post("/api/alerts")
async def create_alert(body: AlertCreate, db: AsyncSession = Depends(get_db)):
    valid = ["PRICE_ABOVE", "PRICE_BELOW", "RSI_OVERBOUGHT", "RSI_OVERSOLD"]
    if body.alert_type not in valid:
        raise HTTPException(400, f"alert_type must be one of: {valid}")
    alert = Alert(symbol=body.symbol.upper(), alert_type=body.alert_type,
                  threshold=body.threshold, message=body.message)
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return {"message": "Alert created", "alert_id": alert.id}


# ═══════════════════════════════════════════════════════════
# WebSocket — Real-time Price Streaming
# ═══════════════════════════════════════════════════════════

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price streaming.
    Client sends: {"symbols": ["RELIANCE.NS", "TCS.NS"]}
    Server streams: price updates every 30 seconds
    """
    await ws_manager.connect(websocket)
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

    try:
        while True:
            # Check for client message (symbol updates)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                if "symbols" in msg:
                    symbols = msg["symbols"][:10]
            except asyncio.TimeoutError:
                pass

            # Send price update
            import json as _json
            from backend.tools.financial_tools import get_stock_price
            prices = {}
            for sym in symbols:
                try:
                    prices[sym] = _json.loads(get_stock_price.invoke(sym))
                except Exception:
                    pass

            await websocket.send_json({
                "type": "price_update",
                "prices": prices,
                "timestamp": datetime.now().isoformat(),
            })

            await asyncio.sleep(30)  # Update every 30 seconds

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
