"""
Financial Agent (LLM Router) — LangGraph + Groq + MCP Server
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

from components.mcp_manager import MCPToolManager
from components.mcp_manager_sse import MCPToolManagerSSE
from components.agent_router import FinancialAgentRouter

load_dotenv()

mcp_managers: list[MCPToolManager] = []
agent: Optional[FinancialAgentRouter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_managers, agent
    print("_" * 52)
    print("  Financial Agent  |  LLM Router + MCP Server")
    print("_" * 52)

    # 1. Start stock market MCP server
    mcp_managers.append(MCPToolManager(server_script="./stock_market_server.py"))

    # 2. Start Alpha Vantage MCP server (if API KEY is present)
    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if av_api_key:
        try:
            alpha_mgr = MCPToolManager(
                command="uvx",
                args=["--from", "marketdata-mcp-server", "marketdata-mcp", av_api_key],
            )
            await alpha_mgr.__aenter__()
            if len(alpha_mgr.langchain_tools) == 0:
                await alpha_mgr.__aexit__(None, None, None)
                raise RuntimeError("Alpha Vantage stdio returned 0 tools")
            alpha_mgr._already_entered = True
            mcp_managers.append(alpha_mgr)
        except Exception as exc:
            print(f"[warn] Alpha Vantage stdio failed ({exc}). Falling back to SSE...")
            alpha_url = f"https://mcp.alphavantage.co/mcp?apikey={av_api_key}"
            mcp_managers.append(MCPToolManagerSSE(alpha_url))
    else:
        print("[warn] ALPHA_VANTAGE_API_KEY not found. Skipping Alpha Vantage tools.")

    for mgr in mcp_managers:
        if mgr is None:
            continue
        if getattr(mgr, "_already_entered", False):
            continue
        await mgr.__aenter__()

    agent = FinancialAgentRouter(mcp_managers)
    print(f"[ready] Total tools loaded: {len(agent.tools)}")

    yield

    print("[shutdown] Stopping MCP servers...")
    for mgr in reversed(mcp_managers):
        await mgr.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)

class NewsRequest(BaseModel):
    query: str
    reset: bool = False

@app.post("/api/news")
async def api_news(req: NewsRequest):
    if req.reset and agent is not None:
        agent.reset()

    if agent is None:
        return {"error": "Agent is not initialized yet."}

    try:
        result = await agent.chat(req.query)
        return result
    except Exception as exc:
        return {"error": str(exc)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
