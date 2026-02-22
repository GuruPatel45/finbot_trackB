"""
backend/agents/langgraph_agent.py
===================================
Financial Research Agent built with LangGraph.

LangGraph Workflow (Multi-Node):
  ┌─────────────────────────────────────────────────────┐
  │                   START                             │
  │                     ↓                               │
  │            [classify_query]                         │
  │          Determines query type:                     │
  │   stock_analysis | portfolio | news | tax | sip     │
  │                     ↓                               │
  │  ┌──────────┬───────┴──────┬──────────┬─────────┐  │
  │  ↓          ↓              ↓          ↓         ↓  │
  │ [analyze]  [portfolio]  [news_hub]  [calc]  [general]│
  │  ↓          ↓              ↓          ↓         ↓  │
  │  └──────────┴───────────────┴──────────┴─────────┘  │
  │                     ↓                               │
  │             [synthesize]                            │
  │   Combines all tool results into final response     │
  │                     ↓                               │
  │                    END                              │
  └─────────────────────────────────────────────────────┘

Why LangGraph over basic LangChain?
  - Explicit node-based workflow (easier to debug + monitor)
  - Conditional routing based on query type
  - State management across nodes
  - Better error recovery (retry specific nodes)
  - Visual graph representation for presentations
"""

import json
import logging
import time
import os
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import pytz

from tools.financial_tools import ALL_TOOLS
from tools.sentiment_tools import SENTIMENT_TOOLS
from config import settings, indian_market

logger = logging.getLogger(__name__)

ALL_AGENT_TOOLS = ALL_TOOLS + SENTIMENT_TOOLS


# ═══════════════════════════════════════════════════════════
# Graph State
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State passed between LangGraph nodes."""
    messages: Annotated[List, add_messages]      # Full conversation history
    query: str                                    # Original user query
    query_type: str                               # classified type
    tool_results: dict                            # Collected tool outputs
    final_response: str                           # Synthesized final answer
    nodes_visited: List[str]                      # For monitoring/LangSmith
    execution_time_ms: float
    error: Optional[str]


# ═══════════════════════════════════════════════════════════
# LLM Setup
# ═══════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.1):
    """Get LLM with Groq primary, OpenAI/Anthropic fallback."""
    if settings.groq_api_key:
        try:
            return ChatGroq(
                model=settings.groq_model,
                groq_api_key=settings.groq_api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, temperature=temperature)
        except Exception as e:
            logger.warning(f"OpenAI fallback failed: {e}")

    if settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=settings.anthropic_api_key, temperature=temperature)
        except Exception as e:
            logger.warning(f"Anthropic fallback failed: {e}")

    raise RuntimeError(
        "No LLM configured. Set GROQ_API_KEY in .env\n"
        "Free key: https://console.groq.com/"
    )


# ═══════════════════════════════════════════════════════════
# System Prompts per Node
# ═══════════════════════════════════════════════════════════

def market_context() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    is_open = (now.weekday() < 5 and
               now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30))
    return f"""Current Date: {now.strftime('%A, %d %B %Y %I:%M %p IST')}
NSE Market: {'🟢 OPEN' if is_open else '🔴 CLOSED'}
Indian Market Hours: 9:15 AM – 3:30 PM IST, Mon–Fri
NSE Suffix: .NS | BSE Suffix: .BO"""


SYSTEM_BASE = """You are FinBot, an expert AI financial research assistant for Indian (NSE/BSE) and global markets.

{market_ctx}

CRITICAL RULES:
1. Always end with: "⚠️ Educational analysis only. Not SEBI-registered investment advice. Consult a registered financial advisor."
2. Use INR (₹) for Indian stocks
3. Never guarantee returns — say "historically", "may", "potential"
4. For Indian stocks, use .NS suffix (RELIANCE.NS, TCS.NS, HDFCBANK.NS)
"""

CLASSIFY_PROMPT = """Classify the user's financial query into ONE category:

Categories:
- stock_analysis: Technical/fundamental stock analysis, price queries, stock comparison
- portfolio: Portfolio risk, MPT analysis, holdings management, diversification
- news_sentiment: News, sentiment, market mood, recent developments
- calculation: SIP calculator, tax (LTCG/STCG), returns calculation
- sector: Sector comparison, industry analysis
- general: General market concepts, explanations, other

User query: {query}

Respond with ONLY the category name, nothing else."""


# ═══════════════════════════════════════════════════════════
# Node Functions
# ═══════════════════════════════════════════════════════════

def classify_query_node(state: AgentState) -> AgentState:
    """Node 1: Classify query type to route to appropriate analysis node."""
    llm = get_llm(temperature=0)
    response = llm.invoke([HumanMessage(content=CLASSIFY_PROMPT.format(query=state["query"]))])
    query_type = response.content.strip().lower()

    valid_types = {"stock_analysis", "portfolio", "news_sentiment", "calculation", "sector", "general"}
    if query_type not in valid_types:
        query_type = "general"

    logger.info(f"Query classified as: {query_type}")
    return {
        **state,
        "query_type": query_type,
        "nodes_visited": state.get("nodes_visited", []) + ["classify"],
    }


def stock_analysis_node(state: AgentState) -> AgentState:
    """Node 2a: Deep stock analysis using technical + fundamental + sentiment tools."""
    llm = get_llm().bind_tools(ALL_AGENT_TOOLS)

    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
STOCK ANALYSIS MODE:
- Use get_stock_price to fetch current price
- Use get_technical_indicators for RSI, MACD, Bollinger Bands
- Use get_fundamental_analysis for P/E, ROE, margins
- Use analyze_news_sentiment for recent news
- Combine all data into a comprehensive BUY/HOLD/SELL analysis
- Include entry price zones, support/resistance levels
- Always mention risk factors
"""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ]

    # Agentic loop — keep calling tools until done
    tool_results = {}
    for _ in range(6):  # max 6 tool calls
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute tool calls
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            tool_fn = next((t for t in ALL_AGENT_TOOLS if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                    tool_results[tool_name] = result
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")

    return {
        **state,
        "messages": messages,
        "tool_results": {**state.get("tool_results", {}), **tool_results},
        "final_response": final_text,
        "nodes_visited": state.get("nodes_visited", []) + ["stock_analysis"],
    }


def portfolio_node(state: AgentState) -> AgentState:
    """Node 2b: Portfolio risk analysis and optimization."""
    llm = get_llm().bind_tools([t for t in ALL_AGENT_TOOLS if t.name in
                                 ["analyze_portfolio_risk", "get_stock_price", "compare_stocks"]])

    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
PORTFOLIO MODE:
- Use analyze_portfolio_risk for MPT metrics (Sharpe, VaR, correlation)
- Assess diversification across sectors
- Provide specific rebalancing suggestions
- Calculate overall P&L if holding prices given
- Reference Indian market sectors when relevant
"""
    messages = [SystemMessage(content=system), HumanMessage(content=state["query"])]
    tool_results = {}

    for _ in range(4):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tool_fn = next((t for t in ALL_AGENT_TOOLS if t.name == tc["name"]), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                    tool_results[tc["name"]] = result
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")
    return {**state, "messages": messages, "tool_results": {**state.get("tool_results", {}), **tool_results},
            "final_response": final_text, "nodes_visited": state.get("nodes_visited", []) + ["portfolio"]}


def news_sentiment_node(state: AgentState) -> AgentState:
    """Node 2c: News and sentiment analysis."""
    llm = get_llm().bind_tools([t for t in ALL_AGENT_TOOLS if t.name in
                                  ["analyze_news_sentiment", "get_market_overview"]])

    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
NEWS & SENTIMENT MODE:
- Use analyze_news_sentiment for stock-specific news
- Use get_market_overview for broad market sentiment
- Summarize key themes and market implications
- Identify potential price catalysts
- Rate overall sentiment: BULLISH / BEARISH / NEUTRAL
"""
    messages = [SystemMessage(content=system), HumanMessage(content=state["query"])]
    tool_results = {}

    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tool_fn = next((t for t in ALL_AGENT_TOOLS if t.name == tc["name"]), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                    tool_results[tc["name"]] = result
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")
    return {**state, "messages": messages, "tool_results": {**state.get("tool_results", {}), **tool_results},
            "final_response": final_text, "nodes_visited": state.get("nodes_visited", []) + ["news_sentiment"]}


def calculation_node(state: AgentState) -> AgentState:
    """Node 2d: SIP, LTCG/STCG tax calculations."""
    llm = get_llm().bind_tools([t for t in ALL_AGENT_TOOLS if t.name in
                                  ["calculate_sip", "calculate_indian_tax"]])

    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
CALCULATION MODE:
- Use calculate_sip for SIP/investment projections
- Use calculate_indian_tax for LTCG/STCG tax
- Show year-wise breakdown where applicable
- Explain Indian tax rules clearly (STCG 15%, LTCG 10% above ₹1L)
- Always add CA consultation disclaimer
"""
    messages = [SystemMessage(content=system), HumanMessage(content=state["query"])]
    tool_results = {}

    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tool_fn = next((t for t in ALL_AGENT_TOOLS if t.name == tc["name"]), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                    tool_results[tc["name"]] = result
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")
    return {**state, "messages": messages, "tool_results": {**state.get("tool_results", {}), **tool_results},
            "final_response": final_text, "nodes_visited": state.get("nodes_visited", []) + ["calculation"]}


def sector_node(state: AgentState) -> AgentState:
    """Node 2e: Sector analysis."""
    llm = get_llm().bind_tools([t for t in ALL_AGENT_TOOLS if t.name in ["compare_sector", "get_stock_price"]])

    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
SECTOR ANALYSIS MODE:
- Use compare_sector to get sector-wide data
- Identify sector leaders and laggards
- Discuss macroeconomic tailwinds/headwinds
- Compare sector valuation (P/E vs historical)
"""
    messages = [SystemMessage(content=system), HumanMessage(content=state["query"])]
    tool_results = {}

    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tool_fn = next((t for t in ALL_AGENT_TOOLS if t.name == tc["name"]), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                    tool_results[tc["name"]] = result
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))

    final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content), "")
    return {**state, "messages": messages, "tool_results": {**state.get("tool_results", {}), **tool_results},
            "final_response": final_text, "nodes_visited": state.get("nodes_visited", []) + ["sector"]}


def general_node(state: AgentState) -> AgentState:
    """Node 2f: General financial queries without specific tools."""
    llm = get_llm()
    system = SYSTEM_BASE.format(market_ctx=market_context()) + """
GENERAL MODE:
- Answer financial concepts clearly
- Explain Indian market mechanics, SEBI regulations
- Provide educational content
"""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["query"])])
    return {**state, "final_response": response.content,
            "nodes_visited": state.get("nodes_visited", []) + ["general"]}


# ── Router ────────────────────────────────────────────────

def route_query(state: AgentState) -> str:
    """Conditional edge: route to appropriate analysis node."""
    routes = {
        "stock_analysis": "stock_analysis",
        "portfolio": "portfolio",
        "news_sentiment": "news_sentiment",
        "calculation": "calculation",
        "sector": "sector",
        "general": "general",
    }
    return routes.get(state.get("query_type", "general"), "general")


# ═══════════════════════════════════════════════════════════
# Build LangGraph
# ═══════════════════════════════════════════════════════════

def build_financial_graph() -> StateGraph:
    """Construct the LangGraph financial research workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_query_node)
    graph.add_node("stock_analysis", stock_analysis_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("news_sentiment", news_sentiment_node)
    graph.add_node("calculation", calculation_node)
    graph.add_node("sector", sector_node)
    graph.add_node("general", general_node)

    # Edges
    graph.add_edge(START, "classify")

    # Conditional routing from classify
    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "stock_analysis": "stock_analysis",
            "portfolio": "portfolio",
            "news_sentiment": "news_sentiment",
            "calculation": "calculation",
            "sector": "sector",
            "general": "general",
        }
    )

    # All analysis nodes → END
    for node in ["stock_analysis", "portfolio", "news_sentiment", "calculation", "sector", "general"]:
        graph.add_edge(node, END)

    return graph.compile()


# ── Main Agent Class ──────────────────────────────────────

class FinBotAgent:
    """
    Main agent class wrapping the LangGraph workflow.
    Provides simple .run() interface for API/frontend use.
    """

    def __init__(self):
        self.graph = build_financial_graph()
        self._conversation_history = []
        logger.info(f"FinBotAgent initialized | LLM: {settings.groq_model} via Groq | Tools: {len(ALL_AGENT_TOOLS)}")

    def run(self, query: str, session_id: str = "default") -> dict:
        """
        Run financial research query through LangGraph.

        Args:
            query: User's financial question
            session_id: For conversation tracking

        Returns:
            dict with response, nodes_visited, tools_used, execution_time_ms
        """
        start = time.time()

        # Include conversation history in messages
        messages = self._conversation_history[-10:] + [HumanMessage(content=query)]

        initial_state: AgentState = {
            "messages": messages,
            "query": query,
            "query_type": "",
            "tool_results": {},
            "final_response": "",
            "nodes_visited": [],
            "execution_time_ms": 0.0,
            "error": None,
        }

        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"run_name": f"finbot_{session_id}", "tags": [session_id]},
            )

            response = final_state.get("final_response", "")

            # Add disclaimer if missing
            if "⚠️" not in response:
                response += "\n\n⚠️ Educational analysis only. Not SEBI-registered investment advice. Consult a registered financial advisor."

            # Update conversation memory
            self._conversation_history.append(HumanMessage(content=query))
            self._conversation_history.append(AIMessage(content=response))
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            exec_ms = (time.time() - start) * 1000
            return {
                "response": response,
                "query_type": final_state.get("query_type", ""),
                "nodes_visited": final_state.get("nodes_visited", []),
                "tools_used": list(final_state.get("tool_results", {}).keys()),
                "execution_time_ms": round(exec_ms, 2),
                "status": "success",
            }

        except Exception as e:
            exec_ms = (time.time() - start) * 1000
            logger.error(f"LangGraph error: {e}")
            return {
                "response": f"Error processing query: {str(e)}\n\nPlease check your API keys and try again.",
                "query_type": "error",
                "nodes_visited": [],
                "tools_used": [],
                "execution_time_ms": round(exec_ms, 2),
                "status": "error",
                "error": str(e),
            }

    def reset(self):
        """Clear conversation history."""
        self._conversation_history = []

    def get_graph_diagram(self) -> str:
        """Get ASCII representation of the graph for debugging."""
        return """
LangGraph Financial Research Workflow:
───────────────────────────────────────
START
  │
  ▼
[classify_query] ── Determines query type
  │
  ├── stock_analysis ──► [stock_analysis_node] ─ price + technicals + fundamentals + news
  ├── portfolio       ──► [portfolio_node]      ─ MPT risk + Sharpe + VaR + diversification
  ├── news_sentiment  ──► [news_sentiment_node] ─ NewsAPI + VADER/FinBERT sentiment
  ├── calculation     ──► [calculation_node]    ─ SIP + LTCG/STCG tax calculator
  ├── sector          ──► [sector_node]         ─ IT/Banking/FMCG/Auto/Pharma/Energy compare
  └── general         ──► [general_node]        ─ financial concepts, market education
                                │
                               END
───────────────────────────────────────
"""


# ── Singleton factory ─────────────────────────────────────
_agent_instance = None

def get_agent() -> FinBotAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FinBotAgent()
    return _agent_instance
