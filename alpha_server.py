import os
import requests
import json
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("AlphaVantage")

# Get API key from environment
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

@mcp.tool()
def get_alpha_vantage_quote(symbol: str) -> str:
    """
    Get the latest price and volume information for a stock symbol from Alpha Vantage.

    Args:
        symbol: The stock symbol (e.g., AAPL)
    """
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY not set."

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            if not quote:
                  return f"No quote found for {symbol}."
            return json.dumps(quote, indent=2)
        else:
            return f"Error: Could not retrieve data for {symbol}. {data.get('Note', data.get('Information', ''))}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

@mcp.tool()
def get_alpha_vantage_overview(symbol: str) -> str:
    """
    Get fundamental information about a company from Alpha Vantage.
    Includes sector, industry, market cap, P/E ratio, dividend yield, etc.

    Args:
        symbol: The stock symbol (e.g., MSFT)
    """
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY not set."

    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if not data:
            return f"No overview found for {symbol}."
            
        if "Symbol" in data:
            return json.dumps(data, indent=2)
        else:
             return f"Error: Could not retrieve data for {symbol}. {data.get('Note', data.get('Information', ''))}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

@mcp.tool()
def get_alpha_vantage_daily_series(symbol: str, outputsize: str = "compact") -> str:
    """
    Get daily historical time series (open, high, low, close, volume) for a stock symbol.

    Args:
        symbol: The stock symbol (e.g., TSLA)
        outputsize: 'compact' returns last 100 data points; 'full' returns full digital history (20+ years).
    """
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY not set."

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            # We only return the last 10 days to keep the context window reasonable
            all_days = data["Time Series (Daily)"]
            last_10_days = {date: all_days[date] for date in list(all_days.keys())[:10]}
            return json.dumps({
                "symbol": symbol,
                "data": last_10_days
            }, indent=2)
        else:
             return f"Error: Could not retrieve data for {symbol}. {data.get('Note', data.get('Information', ''))}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')