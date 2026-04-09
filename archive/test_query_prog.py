import asyncio
import os
import json
from contextlib import AsyncExitStack
from test_news_client import MCPToolManager, FinancialAgent

async def test_agent():
    print("Initializing servers...")
    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    mcp1 = MCPToolManager(command="python", args=["./stock_market_server.py"], env=None)
    mcp2 = MCPToolManager(command="uvx", args=["--from", "marketdata-mcp-server", "marketdata-mcp", av_api_key], env=None)
    
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp1)
        await stack.enter_async_context(mcp2)
        
        all_tools = mcp1.langchain_tools + mcp2.langchain_tools
        print(f"Loaded {len(all_tools)} tools. Firing query...")
        
        agent = FinancialAgent(tools=all_tools)
        
        query = "What is the currency exchange rate of USD to INR (US Dollars to Indian Rupee)? Please use the market data tools."
        result = await agent.chat(query)
        
        print("\n=== AGENT RESPONSE ===\n")
        print(json.dumps(result, indent=2))
        print("\n======================\n")

if __name__ == "__main__":
    asyncio.run(test_agent())
