"""
Financial Agent — LangGraph + Groq + MCP Server
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

from components.mcp_manager import MCPToolManager
from components.agent import FinancialAgent

load_dotenv()

# ─────────────────────────────────────────────
# FASTAPI SERVER ENTRY POINT
# ─────────────────────────────────────────────

mcp_managers: list[MCPToolManager] = []
agent: Optional[FinancialAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_managers, agent
    print("━" * 52)
    print("  📈 Financial Agent  |  LangGraph + Groq + MCP Server")
    print("━" * 52)
    
    # 1. Start stock market MCP server
    mcp_managers.append(MCPToolManager(server_script="./stock_market_server.py"))
    
    # 2. Start Alpha Vantage MCP server (if API KEY is present)
    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if av_api_key:
        mcp_managers.append(MCPToolManager(
            command="uvx",
            args=["--from", "marketdata-mcp-server", "marketdata-mcp", av_api_key]
        ))
    else:
        print("⚠️ ALPHA_VANTAGE_API_KEY not found. Skipping Alpha Vantage tools.")
        
    for mgr in mcp_managers:
        await mgr.__aenter__()

    # 3. Verify core environment variables
    required_vars = ["FINNHUB_API_KEY", "GROQ_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ CRITICAL ERROR: {var} environment variable is not set!")
    
    agent = FinancialAgent(mcp_managers)
    print(f"✅ Application is fully ready. Total tools loaded: {len(agent.tools)}")
    
    yield
    
    print("🛑 Shutting down server...")
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


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)