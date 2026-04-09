"""
FastAPI server wrapping the same FinancialAgent logic as test_news_client.py.

Usage:
    python api_server.py                 # starts on http://127.0.0.1:8000
    python api_server.py --port 9000     # custom port

Do NOT use `uvicorn ... --reload` on Windows.
"""

from __future__ import annotations

import os
import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

# ── Windows subprocess fix (MUST be before any loop) ──
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ArticleSummary(BaseModel):
    headline: str = Field(..., description="Article headline")
    source: str = Field(..., description="Publisher/source name")
    date: str = Field(..., description="YYYY-MM-DD date or 'N/A'")
    link: str = Field(..., description="Source URL or 'N/A'")
    summary: str = Field(..., description="1-3 sentence summary tailored to the query")
    relevance: str = Field(..., description="Short clause explaining relevance")

class NewsSummaryResponse(BaseModel):
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="2-4 sentence synthesized answer")
    top_articles: list[ArticleSummary] = Field(..., description="Top 2-5 most relevant articles")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the agent")
    reset: bool = Field(False, description="Clear conversation history before this request")
    keep_history: bool = Field(False, description="Persist conversation history across requests")

class HealthResponse(BaseModel):
    status: str = "ok"
    tools: list[str] = []


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analysis assistant with access to real-time market data and news.

## How you work
1. Receive a user query
2. Fetch relevant data using tools (prices, news, articles, financials)
3. READ and REASON over what the tools return
4. Answer the query in your own words - synthesized, clear, and direct

## Output rules
- NEVER dump raw tool output or JSON at the user.
- For news: fetch articles, scrape top 2-3, synthesize a direct answer.
- For prices: present numbers cleanly with brief interpretation.
- Always ground your answer in what the tools returned.

## News workflow
1. Call get_market_news or get_company_news
2. Call scrape_article on the most relevant articles (top 2-3)
3. Synthesize the scraped content into a direct answer

## Other rules
- For ambiguous tickers, resolve to symbol first
- If a tool fails, try an alternative
"""


# ─────────────────────────────────────────────
# SCHEMA / TOOL UTILS
# ─────────────────────────────────────────────

_JSON_TO_PYTHON = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}

def _build_args_schema(name: str, schema: dict) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for k, v in props.items():
        pt = _JSON_TO_PYTHON.get(v.get("type", "string"), str)
        desc = v.get("description", "")
        fields[k] = (pt, Field(..., description=desc)) if k in required else (Optional[pt], Field(None, description=desc))
    return create_model(name, **fields)


def _format_tool_result(tool_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()
    if tool_name in ("get_market_news", "get_company_news") and isinstance(data, list):
        lines = [f"Found {len(data)} articles:\n"]
        for i, a in enumerate(data[:10], 1):
            h = a.get("headline") or a.get("title", "No title")
            lines.append(f"{i}. [{h}]")
            for k in ("source", "datetime", "summary", "url"):
                if a.get(k): lines.append(f"   {k.title()}: {a[k]}")
            lines.append("")
        return "\n".join(lines)
    if tool_name == "scrape_article" and isinstance(data, dict):
        parts = []
        if data.get("title"):  parts.append(f"Title: {data['title']}")
        if data.get("source"): parts.append(f"Source: {data['source']}")
        c = data.get("content") or data.get("text") or data.get("body", "")
        if c: parts.append(f"\n{c.strip()}")
        return "\n".join(parts) if parts else raw
    if tool_name == "get_stock_price" and isinstance(data, dict):
        return "\n".join(f"{k}: {v}" for k, v in data.items())
    return json.dumps(data, indent=2)


def _make_tool_fn(session: ClientSession, tool_name: str):
    async def _invoke(**kwargs) -> str:
        try:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            result = await session.call_tool(tool_name, clean)
            if not result.content:
                return f"[{tool_name}] no content."
            raw = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text").strip()
            return _format_tool_result(tool_name, raw)
        except Exception as e:
            return f"[{tool_name}] error: {e}"
    _invoke.__name__ = tool_name
    return _invoke


def _wrap_tools(session: ClientSession, mcp_tools) -> list[StructuredTool]:
    wrapped = []
    for t in mcp_tools:
        schema = t.inputSchema or {}
        wrapped.append(StructuredTool(
            name=t.name,
            description=t.description or t.name,
            args_schema=_build_args_schema(f"{t.name}Schema", schema),
            coroutine=_make_tool_fn(session, t.name),
        ))
    return wrapped


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

_NEWS_TOOLS = {"get_market_news", "get_company_news", "scrape_article"}

class Agent:
    def __init__(self, tools: list[StructuredTool], model: str = "openai/gpt-oss-120b"):
        self.tools = tools
        base = ChatGroq(model=model, temperature=0.0)
        self.llm = base.bind_tools(tools)
        self.structured_llm = base.with_structured_output(NewsSummaryResponse)
        self._graph = self._compile()
        self._history: list[BaseMessage] = []

    def _compile(self):
        tn = ToolNode(self.tools)
        def agent_node(state: MessagesState) -> dict:
            msgs = state["messages"]
            if not any(isinstance(m, SystemMessage) for m in msgs):
                msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
            return {"messages": [self.llm.invoke(msgs)]}
        g = StateGraph(MessagesState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tn)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile()

    async def chat(self, user_input: str) -> dict:
        self._history.append(HumanMessage(content=user_input))
        result = await self._graph.ainvoke({"messages": self._history})
        self._history = list(result["messages"])
        news_used = any(isinstance(m, ToolMessage) and m.name in _NEWS_TOOLS for m in self._history)
        if not news_used:
            return {"type": "financial", "query": user_input, "answer": self._text()}
        try:
            s = await self._news_structured(user_input)
            return {"type": "news", "query": s.query, "answer": s.answer, "top_articles": [a.model_dump() for a in s.top_articles]}
        except Exception as e:
            logger.warning("Structured extraction failed: %s", e)
            return {"type": "financial", "query": user_input, "answer": self._text()}

    async def _news_structured(self, user_input: str) -> NewsSummaryResponse:
        msgs = [SystemMessage(content=SYSTEM_PROMPT), *self._history, HumanMessage(
            content=f'The user asked: "{user_input}"\n\nUsing ONLY the tool data, fill the NewsSummaryResponse schema. Be precise. Do not invent info.'
        )]
        return await self.structured_llm.ainvoke(msgs)

    def _text(self) -> str:
        for m in reversed(self._history):
            if isinstance(m, AIMessage) and m.content:
                c = m.content
                if isinstance(c, str): return c
                if isinstance(c, list):
                    t = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text").strip()
                    if t: return t
        return "(no response)"

    def reset(self):
        self._history = []


# ─────────────────────────────────────────────
# LIFESPAN (everything in ONE coroutine scope)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    server_script = os.getenv("MCP_SERVER_SCRIPT") or str(
        Path(__file__).resolve().parent / "stock_market_server.py"
    )

    command = sys.executable if server_script.endswith(".py") else "node"
    params = StdioServerParameters(command=command, args=[server_script], env=None)

    # Everything below stays in THIS coroutine — no cancel-scope cross-task issues
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()
            tools = _wrap_tools(session, response.tools)
            tool_names = [t.name for t in tools]
            logger.info("MCP ready - %d tools: %s", len(tool_names), tool_names)

            app.state.tools = tools
            app.state.tool_names = tool_names
            app.state.agent = Agent(tools)
            app.state.semaphore = asyncio.Semaphore(
                int(os.getenv("NEWS_API_MAX_CONCURRENT", "1"))
            )

            yield  # ← server runs here; MCP stays alive until shutdown


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Financial News Agent API",
    version="1.0.0",
    description="LangGraph + Groq + MCP",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", tools=getattr(app.state, "tool_names", []))


@app.post("/api/news")
async def news_query(payload: QueryRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(400, "query must not be empty")

    base_agent: Agent = app.state.agent
    if payload.keep_history:
        agent = base_agent
        if payload.reset: agent.reset()
    else:
        agent = Agent(app.state.tools)

    retries = int(os.getenv("NEWS_API_MAX_RETRIES", "2"))
    async with app.state.semaphore:
        for attempt in range(retries + 1):
            try:
                return await agent.chat(query)
            except Exception as exc:
                msg = str(exc).lower()
                if "request too large" in msg or "context length" in msg:
                    agent.reset()
                    if attempt < retries: continue
                    raise HTTPException(413, "Request too large.")
                if "rate_limit" in msg or "rate limit" in msg or "429" in msg:
                    if attempt < retries:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        continue
                    raise HTTPException(429, "Rate limit. Retry later.")
                logger.exception("Error attempt %d", attempt)
                raise HTTPException(500, str(exc))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    print("=" * 52)
    print("  Financial News Agent API")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 52)

    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False, loop="asyncio")
