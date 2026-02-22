'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { api, createChatStream, createPriceWebSocket, fmt } from '@/lib/api'
import type { StockPrice, Holding, PortfolioResponse } from '@/lib/api'

// ── Icons (inline SVG to avoid import issues) ─────────────
const Icon = {
  TrendingUp: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>,
  Bot: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2" /></svg>,
  Briefcase: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>,
  Activity: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  Send: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>,
  Plus: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>,
  Trash: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
  Refresh: () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
  Zap: () => <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>,
}

// ── Tabs ─────────────────────────────────────────────────
const TABS = [
  { id: 'chat', label: 'AI Chat', icon: Icon.Bot },
  { id: 'stocks', label: 'Stock Analysis', icon: Icon.TrendingUp },
  { id: 'portfolio', label: 'Portfolio', icon: Icon.Briefcase },
  { id: 'monitor', label: 'Monitoring', icon: Icon.Activity },
]

const WATCHLIST = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS']

const QUICK_QUERIES = [
  'Analyze TCS.NS — technical + fundamental',
  'Compare RELIANCE vs ONGC 1 year',
  'HDFC Bank news sentiment today',
  'Calculate SIP ₹5000/month at 12% for 10 years',
  'IT sector comparison — best stocks',
  'Nifty market overview today',
]

// ═══════════════════════════════════════════════════════════
// Components
// ═══════════════════════════════════════════════════════════

function PriceCard({ symbol, data }: { symbol: string; data?: StockPrice }) {
  if (!data) return (
    <div className="card animate-pulse">
      <div className="h-4 bg-dark-600 rounded w-24 mb-2" />
      <div className="h-6 bg-dark-600 rounded w-32" />
    </div>
  )
  const up = (data.change_pct ?? 0) >= 0
  const name = symbol.replace('.NS', '').replace('.BO', '')
  return (
    <div className="card hover:border-[#2a3a5a] transition-colors cursor-pointer">
      <div className="flex justify-between items-start mb-1">
        <span className="text-xs font-mono text-gray-400">{name}</span>
        <span className={`text-xs font-semibold ${up ? 'text-brand-green' : 'text-brand-red'}`}>
          {up ? '▲' : '▼'} {Math.abs(data.change_pct ?? 0).toFixed(2)}%
        </span>
      </div>
      <div className="text-xl font-bold text-white">{fmt.inr(data.price)}</div>
      <div className="text-xs text-gray-500 mt-1">Vol: {((data.volume ?? 0) / 1e5).toFixed(1)}L</div>
    </div>
  )
}

function ChatMessage({ msg }: { msg: { role: string; content: string; nodes?: string[]; tools?: string[]; ms?: number } }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
        isUser
          ? 'bg-brand-blue/20 border border-brand-blue/30 text-white rounded-br-sm'
          : 'bg-dark-700 border border-[#1a2545] text-gray-200 rounded-bl-sm'
      }`}>
        {!isUser && msg.nodes && msg.nodes.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {msg.nodes.map(n => (
              <span key={n} className="badge badge-blue text-[10px]">⬡ {n}</span>
            ))}
            {msg.tools?.map(t => (
              <span key={t} className="badge badge-yellow text-[10px]">⚙ {t.replace('_', ' ')}</span>
            ))}
          </div>
        )}
        <div className="whitespace-pre-wrap">{msg.content}</div>
        {msg.ms && (
          <div className="text-[10px] text-gray-500 mt-2">{msg.ms.toFixed(0)}ms · Groq Llama 3.3 70B</div>
        )}
      </div>
    </div>
  )
}

function NodeTrace({ nodes, tools }: { nodes: string[]; tools: string[] }) {
  if (!nodes.length) return null
  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-dark-800 border-b border-[#1a2545] text-xs overflow-x-auto">
      <span className="text-gray-500 whitespace-nowrap">LangGraph:</span>
      {nodes.map((n, i) => (
        <span key={n} className="flex items-center gap-1">
          {i > 0 && <span className="text-gray-600">→</span>}
          <span className="badge badge-blue">{n}</span>
        </span>
      ))}
      {tools.length > 0 && (
        <>
          <span className="text-gray-600 ml-2">|</span>
          <span className="text-gray-500">Tools:</span>
          {tools.map(t => <span key={t} className="badge badge-yellow">{t}</span>)}
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════
// Main Page
// ═══════════════════════════════════════════════════════════

export default function Home() {
  const [activeTab, setActiveTab] = useState('chat')
  const [prices, setPrices] = useState<Record<string, StockPrice>>({})
  const [messages, setMessages] = useState<any[]>([{
    role: 'assistant',
    content: '👋 Namaste! Main FinBot hoon — aapka AI financial research assistant.\n\nMujhse poochh sakte ho:\n• Indian stocks (NSE/BSE) ka technical + fundamental analysis\n• Portfolio risk assessment (Modern Portfolio Theory)\n• News sentiment (VADER + FinBERT)\n• LTCG/STCG tax calculations\n• SIP calculator with step-up\n• Sector comparison (IT, Banking, FMCG, Auto...)\n\nExample: "Analyze RELIANCE.NS stock" ya "Compare TCS vs Infosys"',
  }])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [lastTrace, setLastTrace] = useState<{ nodes: string[]; tools: string[] }>({ nodes: [], tools: [] })
  const [streamSteps, setStreamSteps] = useState<string[]>([])

  // Stock tab state
  const [stockSymbol, setStockSymbol] = useState('RELIANCE.NS')
  const [stockData, setStockData] = useState<any>(null)
  const [stockLoading, setStockLoading] = useState(false)

  // Portfolio tab state
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null)
  const [addForm, setAddForm] = useState({ symbol: '', quantity: '', price: '' })
  const [portfolioLoading, setPortfolioLoading] = useState(false)

  // Monitor tab
  const [monitorStats, setMonitorStats] = useState<any>(null)

  const chatEndRef = useRef<HTMLDivElement>(null)
  const sessionId = useRef(`session_${Date.now()}`)

  // ── WebSocket price streaming ─────────────────────────
  useEffect(() => {
    const cleanup = createPriceWebSocket(WATCHLIST, (newPrices) => {
      setPrices(newPrices)
    })
    // Also fetch immediately via REST
    api.getWatchlistPrices(WATCHLIST).then(data => setPrices(data.prices)).catch(console.warn)
    return cleanup
  }, [])

  // ── Auto-scroll chat ──────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Chat with SSE streaming ───────────────────────────
  const sendMessage = useCallback(async (query?: string) => {
    const q = (query ?? input).trim()
    if (!q || isLoading) return

    setInput('')
    setIsLoading(true)
    setStreamSteps([])
    setMessages(prev => [...prev, { role: 'user', content: q }])

    let cleanup: (() => void) | null = null
    const steps: string[] = []

    cleanup = createChatStream(q, sessionId.current, (event) => {
      if (event.type === 'node') {
        steps.push(`⬡ ${event.node}`)
        setStreamSteps([...steps])
      } else if (event.type === 'tool') {
        steps.push(`⚙ ${event.tool}`)
        setStreamSteps([...steps])
      } else if (event.type === 'complete') {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: event.response,
          ms: event.execution_time_ms,
          nodes: steps.filter(s => s.startsWith('⬡')).map(s => s.slice(2)),
          tools: steps.filter(s => s.startsWith('⚙')).map(s => s.slice(2)),
        }])
        setStreamSteps([])
        setIsLoading(false)
      } else if (event.type === 'error') {
        setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${event.message}` }])
        setIsLoading(false)
      }
    })

    // Fallback to REST if SSE fails
    setTimeout(async () => {
      if (isLoading) {
        cleanup?.()
        try {
          const result = await api.chat(q, sessionId.current)
          setLastTrace({ nodes: result.nodes_visited, tools: result.tools_used })
          setMessages(prev => [...prev, {
            role: 'assistant', content: result.response,
            ms: result.execution_time_ms,
            nodes: result.nodes_visited, tools: result.tools_used,
          }])
        } catch (e: any) {
          setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
        } finally {
          setIsLoading(false)
        }
      }
    }, 3000)
  }, [input, isLoading])

  // ── Stock Analysis ────────────────────────────────────
  const analyzeStock = async () => {
    if (!stockSymbol.trim()) return
    setStockLoading(true)
    try {
      const sym = stockSymbol.trim().toUpperCase()
      const data = await api.getStock(sym.includes('.') ? sym : sym + '.NS')
      setStockData(data)
    } catch (e: any) {
      alert(`Error: ${e.message}`)
    } finally {
      setStockLoading(false)
    }
  }

  // ── Portfolio ─────────────────────────────────────────
  const loadPortfolio = async () => {
    setPortfolioLoading(true)
    try {
      const data = await api.getPortfolio(1)
      setPortfolio(data)
    } catch (e) {
      console.error(e)
    } finally {
      setPortfolioLoading(false)
    }
  }

  const addHolding = async () => {
    if (!addForm.symbol || !addForm.quantity || !addForm.price) return
    try {
      await api.addHolding(1, {
        symbol: addForm.symbol,
        quantity: parseFloat(addForm.quantity),
        avg_buy_price: parseFloat(addForm.price),
      })
      setAddForm({ symbol: '', quantity: '', price: '' })
      loadPortfolio()
    } catch (e: any) {
      alert(`Error: ${e.message}`)
    }
  }

  const removeHolding = async (id: number) => {
    await api.removeHolding(1, id)
    loadPortfolio()
  }

  useEffect(() => { loadPortfolio() }, [])

  // ── Monitoring ────────────────────────────────────────
  useEffect(() => {
    if (activeTab === 'monitor') {
      api.getStats().then(setMonitorStats).catch(console.warn)
    }
  }, [activeTab])

  // ═════════════════════════════════════════════════════
  // Render
  // ═════════════════════════════════════════════════════

  return (
    <div className="flex h-screen overflow-hidden bg-dark-900">

      {/* ── Sidebar ──────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 bg-dark-800 border-r border-[#1a2545] flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-[#1a2545]">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-blue to-brand-purple flex items-center justify-center">
              <Icon.Zap />
            </div>
            <div>
              <div className="font-bold text-white">FinBot</div>
              <div className="text-[10px] text-gray-500">Track B · LangGraph</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="p-3 flex-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium mb-1 transition-colors ${
                activeTab === tab.id
                  ? 'bg-brand-blue/20 text-brand-blue border border-brand-blue/30'
                  : 'text-gray-400 hover:text-white hover:bg-dark-700'
              }`}
            >
              <tab.icon />
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Live Watchlist */}
        <div className="p-3 border-t border-[#1a2545]">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Live Watchlist</div>
          {WATCHLIST.slice(0, 5).map(sym => {
            const d = prices[sym]
            const name = sym.replace('.NS', '')
            const up = (d?.change_pct ?? 0) >= 0
            return (
              <div key={sym} className="flex justify-between items-center py-1.5 text-xs">
                <span className="font-mono text-gray-300">{name}</span>
                <div className="text-right">
                  {d ? (
                    <>
                      <div className="text-white font-medium">₹{d.price.toLocaleString('en-IN')}</div>
                      <div className={up ? 'text-brand-green' : 'text-brand-red'}>
                        {up ? '▲' : '▼'}{Math.abs(d.change_pct).toFixed(2)}%
                      </div>
                    </>
                  ) : <div className="text-gray-600">Loading...</div>}
                </div>
              </div>
            )
          })}
          <div className="text-[10px] text-gray-600 mt-2 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-green animate-pulse inline-block" />
            WebSocket Live
          </div>
        </div>

        {/* Disclaimer */}
        <div className="p-3 border-t border-[#1a2545]">
          <p className="text-[10px] text-yellow-600/70">
            ⚠️ Educational only. Not SEBI-registered advice.
          </p>
        </div>
      </div>

      {/* ── Main Content ─────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* ── TAB: AI Chat ───────────────────────────── */}
        {activeTab === 'chat' && (
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="px-6 py-4 border-b border-[#1a2545] bg-dark-800 flex items-center justify-between">
              <div>
                <h1 className="font-bold text-white">AI Financial Research Chat</h1>
                <p className="text-xs text-gray-500">LangGraph · Groq Llama 3.3 70B · 10 Tools</p>
              </div>
              <button onClick={() => setMessages([])} className="btn-ghost text-xs">Clear</button>
            </div>

            {/* LangGraph trace bar */}
            {(lastTrace.nodes.length > 0 || streamSteps.length > 0) && (
              <NodeTrace
                nodes={streamSteps.filter(s => s.startsWith('⬡')).map(s => s.slice(2))}
                tools={streamSteps.filter(s => s.startsWith('⚙')).map(s => s.slice(2))}
              />
            )}

            {/* Quick queries */}
            <div className="px-6 py-3 border-b border-[#1a2545] flex gap-2 overflow-x-auto">
              {QUICK_QUERIES.map(q => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-xs whitespace-nowrap px-3 py-1.5 rounded-full bg-dark-700 text-gray-400 hover:text-white hover:bg-dark-600 transition-colors border border-[#1a2545]"
                >
                  {q.length > 30 ? q.slice(0, 30) + '...' : q}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6">
              {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}

              {/* Loading indicator */}
              {isLoading && (
                <div className="flex items-start gap-3 mb-3">
                  <div className="card border-brand-blue/30 max-w-md">
                    {streamSteps.length > 0 ? (
                      <div className="space-y-1">
                        {streamSteps.map((step, i) => (
                          <div key={i} className="text-xs text-gray-400 flex items-center gap-2">
                            <span className="text-brand-blue animate-spin">⟳</span>
                            {step}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex gap-1 items-center">
                        <div className="w-2 h-2 bg-brand-blue rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-brand-blue rounded-full animate-bounce delay-100" />
                        <div className="w-2 h-2 bg-brand-blue rounded-full animate-bounce delay-200" />
                        <span className="text-xs text-gray-500 ml-2">LangGraph processing...</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-[#1a2545] bg-dark-800">
              <div className="flex gap-3">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
                  placeholder="Ask about stocks, portfolio, tax, SIP... (Enter to send)"
                  className="input resize-none"
                  rows={2}
                  disabled={isLoading}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || isLoading}
                  className="btn-primary flex items-center gap-2 px-5 self-end"
                >
                  <Icon.Send />
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB: Stock Analysis ─────────────────────── */}
        {activeTab === 'stocks' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="mb-6">
              <h2 className="text-xl font-bold mb-1">Stock Analysis</h2>
              <p className="text-gray-500 text-sm">Technical + Fundamental analysis for NSE/BSE stocks</p>
            </div>

            <div className="flex gap-3 mb-6">
              <input
                className="input max-w-xs"
                value={stockSymbol}
                onChange={e => setStockSymbol(e.target.value.toUpperCase())}
                placeholder="RELIANCE.NS"
                onKeyDown={e => e.key === 'Enter' && analyzeStock()}
              />
              <button onClick={analyzeStock} disabled={stockLoading} className="btn-primary">
                {stockLoading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>

            {/* Quick symbols */}
            <div className="flex flex-wrap gap-2 mb-6">
              {['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'BAJFINANCE.NS', 'MARUTI.NS'].map(sym => (
                <button key={sym} onClick={() => { setStockSymbol(sym); }}
                  className="text-xs px-3 py-1 rounded-full bg-dark-700 text-gray-400 hover:text-brand-blue border border-[#1a2545] hover:border-brand-blue transition-colors">
                  {sym.replace('.NS', '')}
                </button>
              ))}
            </div>

            {stockData && (
              <div className="space-y-4">
                {/* Price card */}
                <div className="card">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="text-gray-400 text-sm">{stockData.symbol}</div>
                      <div className="text-3xl font-bold mt-1">{fmt.inr(stockData.price?.price ?? 0)}</div>
                    </div>
                    <div className={`text-right ${(stockData.price?.change_pct ?? 0) >= 0 ? 'text-brand-green' : 'text-brand-red'}`}>
                      <div className="text-xl font-semibold">{fmt.pct(stockData.price?.change_pct ?? 0)}</div>
                      <div className="text-sm">{fmt.inr(Math.abs(stockData.price?.change ?? 0))} today</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-[#1a2545]">
                    <div><div className="text-xs text-gray-500">52W High</div><div className="font-medium">{fmt.inr(stockData.price?.high_52w ?? 0)}</div></div>
                    <div><div className="text-xs text-gray-500">52W Low</div><div className="font-medium">{fmt.inr(stockData.price?.low_52w ?? 0)}</div></div>
                    <div><div className="text-xs text-gray-500">Volume</div><div className="font-medium">{((stockData.price?.volume ?? 0) / 1e5).toFixed(1)}L</div></div>
                    <div><div className="text-xs text-gray-500">Market</div><div className={`font-medium text-sm ${stockData.price?.market_status === 'OPEN' ? 'text-brand-green' : 'text-brand-red'}`}>{stockData.price?.market_status}</div></div>
                  </div>
                </div>

                {/* Technical indicators */}
                {stockData.technicals && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="card">
                      <div className="text-sm font-semibold mb-3 text-gray-300">Momentum Indicators</div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-500">RSI (14)</span>
                          <span className={`text-xs font-mono font-medium ${
                            stockData.technicals.momentum?.rsi_14 < 30 ? 'text-brand-green' :
                            stockData.technicals.momentum?.rsi_14 > 70 ? 'text-brand-red' : 'text-white'
                          }`}>{stockData.technicals.momentum?.rsi_14?.toFixed(1)} — {stockData.technicals.momentum?.rsi_zone}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-500">MACD</span>
                          <span className={`text-xs font-mono ${(stockData.technicals.momentum?.macd_line ?? 0) > (stockData.technicals.momentum?.macd_signal ?? 0) ? 'text-brand-green' : 'text-brand-red'}`}>
                            {(stockData.technicals.momentum?.macd_line ?? 0) > (stockData.technicals.momentum?.macd_signal ?? 0) ? '▲ Bullish' : '▼ Bearish'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-gray-500">Stoch K/D</span>
                          <span className="text-xs font-mono text-white">{stockData.technicals.momentum?.stoch_k?.toFixed(1)} / {stockData.technicals.momentum?.stoch_d?.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="card">
                      <div className="text-sm font-semibold mb-3 text-gray-300">Moving Averages</div>
                      <div className="space-y-2">
                        {Object.entries(stockData.technicals.moving_averages ?? {}).slice(0, 4).map(([key, val]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-xs text-gray-500">{key.replace('_', ' ').toUpperCase()}</span>
                            <span className={`text-xs font-mono ${
                              (val as number) < (stockData.price?.price ?? 0) ? 'text-brand-green' : 'text-brand-red'
                            }`}>{fmt.inr(val as number)}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Signals */}
                    <div className="card col-span-2">
                      <div className="text-sm font-semibold mb-3 text-gray-300">
                        Trading Signals
                        <span className={`ml-2 badge ${stockData.technicals.overall_trend === 'BULLISH' ? 'badge-green' : stockData.technicals.overall_trend === 'BEARISH' ? 'badge-red' : 'badge-blue'}`}>
                          {stockData.technicals.overall_trend}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {(stockData.technicals.signals ?? []).map((s: any, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-xs">
                            <span className={s.strength === 'STRONG' ? 'text-brand-yellow' : 'text-gray-500'}>●</span>
                            <span className="text-gray-300">{s.desc}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── TAB: Portfolio ──────────────────────────── */}
        {activeTab === 'portfolio' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-bold">Portfolio Tracker</h2>
                <p className="text-gray-500 text-sm">Live P&L tracking with Indian market data</p>
              </div>
              <button onClick={loadPortfolio} className="btn-ghost flex items-center gap-2 text-sm">
                <Icon.Refresh /> Refresh
              </button>
            </div>

            {/* Summary cards */}
            {portfolio && (
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: 'Invested', val: fmt.inr(portfolio.summary.total_invested) },
                  { label: 'Current Value', val: fmt.inr(portfolio.summary.current_value) },
                  { label: 'Total P&L', val: fmt.inr(portfolio.summary.total_pnl), pct: portfolio.summary.total_pnl_pct, color: portfolio.summary.total_pnl >= 0 },
                  { label: 'Holdings', val: portfolio.summary.holdings_count.toString() },
                ].map((m, i) => (
                  <div key={i} className="card">
                    <div className="text-xs text-gray-500 mb-1">{m.label}</div>
                    <div className={`text-lg font-bold ${m.color === undefined ? 'text-white' : m.color ? 'text-brand-green' : 'text-brand-red'}`}>{m.val}</div>
                    {m.pct !== undefined && (
                      <div className={`text-xs ${m.color ? 'text-brand-green' : 'text-brand-red'}`}>{fmt.pct(m.pct)}</div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Add holding form */}
            <div className="card mb-4">
              <div className="text-sm font-semibold mb-3">Add Holding</div>
              <div className="flex gap-3">
                <input className="input" placeholder="Symbol (e.g. TCS.NS)" value={addForm.symbol}
                  onChange={e => setAddForm({ ...addForm, symbol: e.target.value.toUpperCase() })} />
                <input className="input" type="number" placeholder="Quantity" value={addForm.quantity}
                  onChange={e => setAddForm({ ...addForm, quantity: e.target.value })} />
                <input className="input" type="number" placeholder="Avg Buy Price (₹)" value={addForm.price}
                  onChange={e => setAddForm({ ...addForm, price: e.target.value })} />
                <button onClick={addHolding} className="btn-primary flex items-center gap-2 whitespace-nowrap">
                  <Icon.Plus /> Add
                </button>
              </div>
            </div>

            {/* Holdings table */}
            {portfolioLoading ? (
              <div className="text-center text-gray-500 py-12">Loading portfolio...</div>
            ) : portfolio?.holdings.length === 0 ? (
              <div className="card text-center py-12 text-gray-500">
                No holdings yet. Add stocks above or try the example portfolio.
              </div>
            ) : (
              <div className="card overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[#1a2545] text-gray-500 text-xs uppercase">
                      {['Symbol', 'Qty', 'Buy Price', 'Current', 'Invested', 'Value', 'P&L', 'P&L %', ''].map(h => (
                        <th key={h} className="text-left py-2 px-3 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio?.holdings.map(h => (
                      <tr key={h.id} className="border-b border-[#1a2545]/50 hover:bg-dark-700/30 transition-colors">
                        <td className="py-3 px-3 font-mono text-brand-blue">{h.symbol}</td>
                        <td className="py-3 px-3 text-gray-300">{h.quantity}</td>
                        <td className="py-3 px-3">{fmt.inr(h.avg_buy_price)}</td>
                        <td className="py-3 px-3">
                          <div>{fmt.inr(h.current_price)}</div>
                          <div className={`text-xs ${h.change_pct >= 0 ? 'text-brand-green' : 'text-brand-red'}`}>
                            {fmt.pct(h.change_pct)}
                          </div>
                        </td>
                        <td className="py-3 px-3 text-gray-400">{fmt.inr(h.invested_value)}</td>
                        <td className="py-3 px-3">{fmt.inr(h.current_value)}</td>
                        <td className={`py-3 px-3 font-medium ${h.pnl >= 0 ? 'text-brand-green' : 'text-brand-red'}`}>{fmt.inr(h.pnl)}</td>
                        <td className={`py-3 px-3 ${h.pnl_pct >= 0 ? 'text-brand-green' : 'text-brand-red'}`}>{fmt.pct(h.pnl_pct)}</td>
                        <td className="py-3 px-3">
                          <button onClick={() => removeHolding(h.id)} className="text-gray-600 hover:text-brand-red transition-colors">
                            <Icon.Trash />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── TAB: Monitoring ─────────────────────────── */}
        {activeTab === 'monitor' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="mb-6">
              <h2 className="text-xl font-bold">System Monitoring</h2>
              <p className="text-gray-500 text-sm">LangSmith-style observability for the financial agent</p>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-6">
              {monitorStats ? [
                { label: 'Total Queries', val: monitorStats.total_queries },
                { label: 'Avg Response Time', val: `${monitorStats.avg_execution_time_ms?.toFixed(0)}ms` },
                { label: 'Cache Status', val: monitorStats.cache_status?.redis_available ? '🟢 Redis' : '🟡 In-Memory' },
              ].map((m, i) => (
                <div key={i} className="card">
                  <div className="text-xs text-gray-500">{m.label}</div>
                  <div className="text-2xl font-bold mt-1">{m.val}</div>
                </div>
              )) : <div className="col-span-3 text-center text-gray-500 py-8">Loading stats...</div>}
            </div>

            {/* LangGraph architecture diagram */}
            <div className="card mb-4">
              <div className="text-sm font-semibold mb-4">LangGraph Workflow</div>
              <div className="font-mono text-xs text-gray-400 leading-relaxed">
                <div className="text-brand-blue">START</div>
                <div className="ml-4">↓</div>
                <div className="ml-4 text-brand-yellow">[classify_query] ← Determines query type</div>
                <div className="ml-4">↓</div>
                <div className="ml-4 flex gap-6 flex-wrap">
                  {['stock_analysis', 'portfolio', 'news_sentiment', 'calculation', 'sector', 'general'].map(n => (
                    <span key={n} className="badge badge-blue text-[10px]">⬡ {n}</span>
                  ))}
                </div>
                <div className="ml-4">↓</div>
                <div className="ml-4 text-brand-green">END → Structured JSON response</div>
              </div>
            </div>

            {/* Tech stack */}
            <div className="card">
              <div className="text-sm font-semibold mb-4">Tech Stack (Track B)</div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                {[
                  { label: 'LLM', val: `Groq — ${monitorStats?.llm_model || 'llama-3.3-70b-versatile'}` },
                  { label: 'Agent Framework', val: 'LangGraph (multi-node workflow)' },
                  { label: 'Backend', val: 'FastAPI + asyncio + WebSocket + SSE' },
                  { label: 'Frontend', val: 'Next.js 14 + Tailwind CSS + TypeScript' },
                  { label: 'Database', val: 'PostgreSQL (async SQLAlchemy 2.0)' },
                  { label: 'Cache', val: `Redis + In-memory fallback` },
                  { label: 'Monitoring', val: `LangSmith (${monitorStats?.langsmith_project || 'not configured'})` },
                  { label: 'Financial APIs', val: 'Yahoo Finance + Alpha Vantage + FMP + NewsAPI' },
                  { label: 'Sentiment', val: 'VADER + FinBERT (HuggingFace)' },
                  { label: 'Tools', val: '10 LangChain tools with failover' },
                  { label: 'Portfolio', val: 'Modern Portfolio Theory (MPT) — Sharpe, VaR, CVaR' },
                  { label: 'Deployment', val: 'Vercel (frontend) + Railway (backend)' },
                ].map(({ label, val }) => (
                  <div key={label} className="flex gap-2">
                    <span className="text-gray-500 min-w-[100px]">{label}:</span>
                    <span className="text-white">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
