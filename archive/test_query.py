import asyncio
from gemini_mcp_client import GeminiMCPClient

async def run():
    client = GeminiMCPClient()
    try:
        print("Connecting to server...")
        await client.connect_to_server("./stock_market_server.py")
        print("Connected.")
        
        print("Processing query...")
        response = await client.process_query("What is the current stock price of AAPL?")
        print(f"Final response: \n{response}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(run())
