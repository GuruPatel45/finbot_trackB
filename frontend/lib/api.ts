// frontend/lib/api.ts
// ====================
// Typed API client for the FastAPI backend

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

// ── Types ─────────────────────────────────────────────────

export interface ChatResponse {
  response: string
  query_type: string
  nodes_visited: string[]
  tools_used: string[]
  execution_time_ms: number
  status: string
}

export interface StockPrice {
  symbol: string
  price: number
  currency: string
  change: number
  change_pct: number
  direction: string
  volume: number
  high_52w: number
  low_52w: number
  market_status: string
  timestamp: string
  error?: string
}

export interface TechnicalIndicators {
  symbol: string
  current_price: number
  moving_averages: Record<string, number>
  momentum: {
    rsi_14: number
    rsi_zone: string
    macd_line: number
    macd_signal: number
    macd_histogram: number
    stoch_k: number
    stoch_d: number
  }
  volatility: {
    bb_upper: number
    bb_lower: number
    bb_position_pct: number
    atr_14: number
    atr_pct: number
  }
  volume: { current: number; avg_20d: number; ratio: number; vwap: number }
  overall_trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  signals: Array<{ signal: string; strength: string; desc: string }>
}

export interface Holding {
  id: number
  symbol: string
  exchange: string
  quantity: number
  avg_buy_price: number
  current_price: number
  invested_value: number
  current_value: number
  pnl: number
  pnl_pct: number
  sector?: string
  change_pct: number
}

export interface PortfolioResponse {
  portfolio_id: number
  name: string
  holdings: Holding[]
  summary: {
    total_invested: number
    current_value: number
    total_pnl: number
    total_pnl_pct: number
    holdings_count: number
  }
}

// ── Fetch helper ──────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`API Error ${res.status}: ${err}`)
  }
  return res.json()
}

// ── API Functions ─────────────────────────────────────────

export const api = {
  // Chat
  chat: (query: string, session_id = 'default') =>
    apiFetch<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ query, session_id }),
    }),

  // Stocks
  getStock: (symbol: string, period = '6mo') =>
    apiFetch<{ symbol: string; price: StockPrice; technicals: TechnicalIndicators }>(
      `/api/stocks/${encodeURIComponent(symbol)}?period=${period}`
    ),

  getFundamentals: (symbol: string) =>
    apiFetch<any>(`/api/stocks/${encodeURIComponent(symbol)}/fundamentals`),

  getSector: (sector: string) =>
    apiFetch<any>(`/api/sectors/${sector}`),

  getWatchlistPrices: (symbols: string[]) =>
    apiFetch<{ prices: Record<string, StockPrice>; updated_at: string }>(
      `/api/watchlist/prices?symbols=${symbols.join(',')}`
    ),

  // Portfolio
  getPortfolio: (id = 1) => apiFetch<PortfolioResponse>(`/api/portfolio/${id}`),

  addHolding: (portfolioId: number, data: {
    symbol: string; quantity: number; avg_buy_price: number; exchange?: string; sector?: string
  }) => apiFetch<any>(`/api/portfolio/${portfolioId}/holdings`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  removeHolding: (portfolioId: number, holdingId: number) =>
    apiFetch<any>(`/api/portfolio/${portfolioId}/holdings/${holdingId}`, { method: 'DELETE' }),

  // Alerts
  getAlerts: () => apiFetch<{ alerts: any[] }>('/api/alerts'),
  createAlert: (data: { symbol: string; alert_type: string; threshold: number; message?: string }) =>
    apiFetch<any>('/api/alerts', { method: 'POST', body: JSON.stringify(data) }),

  // Monitoring
  getStats: () => apiFetch<any>('/api/monitoring/stats'),
  health: () => apiFetch<any>('/api/health'),
}

// ── SSE Stream (agent real-time output) ───────────────────

export function createChatStream(
  query: string,
  sessionId = 'default',
  onEvent: (event: { type: string; [key: string]: any }) => void
): () => void {
  const url = `${BASE_URL}/api/chat/stream?query=${encodeURIComponent(query)}&session_id=${sessionId}`
  const es = new EventSource(url)

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent(data)
      if (data.type === 'complete' || data.type === 'error') {
        es.close()
      }
    } catch (err) {
      console.error('SSE parse error:', err)
    }
  }

  es.onerror = () => {
    onEvent({ type: 'error', message: 'Connection failed' })
    es.close()
  }

  return () => es.close()
}

// ── WebSocket Price Stream ─────────────────────────────────

export function createPriceWebSocket(
  symbols: string[],
  onUpdate: (prices: Record<string, StockPrice>) => void
): () => void {
  let ws: WebSocket | null = null
  let reconnectTimer: any = null

  const connect = () => {
    ws = new WebSocket(`${WS_URL}/ws/prices`)

    ws.onopen = () => {
      ws?.send(JSON.stringify({ symbols }))
    }

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        if (data.type === 'price_update') {
          onUpdate(data.prices)
        }
      } catch (err) {
        console.error('WS parse error:', err)
      }
    }

    ws.onclose = () => {
      reconnectTimer = setTimeout(connect, 5000)
    }
  }

  connect()

  return () => {
    clearTimeout(reconnectTimer)
    ws?.close()
  }
}

// ── Formatting helpers ─────────────────────────────────────

export const fmt = {
  inr: (v: number) => `₹${v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
  pct: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`,
  num: (v: number) => v.toLocaleString('en-IN'),
  cr: (v: number) => `₹${(v / 1e7).toFixed(0)} Cr`,
}
