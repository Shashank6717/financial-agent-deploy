"""
Simple HTTP API wrapper for test_news_client.FinancialAgent.

Run:
  uvicorn news_api_server:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from test_news_client import MCPToolManager, FinancialAgent


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the agent")
    reset: bool = Field(False, description="Clear conversation history before running")
    keep_history: bool = Field(False, description="Persist conversation history across requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    server_script = os.getenv("MCP_SERVER_SCRIPT")
    if not server_script:
        server_script = str(Path(__file__).resolve().parent / "stock_market_server.py")

    async with MCPToolManager(server_script) as mcp:
        app.state.mcp = mcp
        app.state.agent = FinancialAgent(mcp)
        yield


app = FastAPI(title="News Agent API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/news")
async def news_query(payload: QueryRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    base_agent: FinancialAgent = app.state.agent
    if payload.keep_history:
        agent = base_agent
        if payload.reset:
            agent.reset()
    else:
        # Default to stateless to avoid token buildup and TPM limits.
        agent = FinancialAgent(app.state.mcp)

    try:
        return await agent.chat(query)
    except Exception as exc:
        msg = str(exc)
        # Handle Groq "request too large" by clearing history and retrying once.
        if "Request too large" in msg or "rate_limit_exceeded" in msg:
            agent.reset()
            return await agent.chat(query)
        raise
