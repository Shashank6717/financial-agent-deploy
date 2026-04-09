import os
import finnhub
import json  # This is a built-in module, no need to add to requirements.txt
import time
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from readability import Document
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env
load_dotenv()

# Initialize FastMCP server                                                                                                                                
mcp = FastMCP("stock-market")

# Get API key from environment                                                                                                                             
api_key = os.getenv('FINNHUB_API_KEY')
if not api_key:
    raise ValueError("FINNHUB_API_KEY environment variable not set")

finnhub_client = finnhub.Client(api_key=api_key)

MAX_ARTICLE_CHARS = 12000

def _extract_readable_text(html: str) -> tuple[str, str]:
    """Extract the main text and title from an HTML document."""
    title = ""
    text = ""
    try:
        doc = Document(html)
        title = doc.short_title() or ""
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:
        # Fallback to raw HTML if readability fails
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = soup.get_text(separator="\n")

    # Normalize whitespace and trim
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS].rstrip() + "..."

    return title, text

@mcp.tool()
def get_stock_symbol_lookup(query: str) -> str:
    """
    Stock Symbol Lookup - searches for best-matching symbols based on your query.

    Args:
        query: You can input anything from symbol, security's name to ISIN and CUSIP e.g. apple

    Returns:
        str: A list of matching symbols with formatted information
    """
    try:
        data = finnhub_client.symbol_lookup(query)

        if not data or "result" not in data:
            return "Unable to look up symbol."

        if len(data["result"]) == 0:
            return f"No symbols found matching '{query}'."

            # Format the response with more readable structure
        formatted_data = {
            "search_query": query,
            "total_matches": data["count"],
            "matches": []
        }

        for item in data["result"]:
            formatted_data["matches"].append({
                "symbol": item["symbol"],
                "name": item["description"],
                "display_symbol": item["displaySymbol"],
                "type": item["type"]
            })

            # Return formatted JSON string
        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error looking up symbol: {str(e)}"


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Get the latest stock price and related information for a given symbol.

    Args:
        symbol: The stock symbol to look up (e.g., AAPL for Apple Inc.)

    Returns:
        str: Current price information in JSON format
    """
    try:
        # Get quote data from Finnhub
        data = finnhub_client.quote(symbol.upper())
        
        if not data or "c" not in data:
            return f"Unable to get price information for symbol '{symbol}'."
            
        if data["c"] == 0 and data["h"] == 0 and data["l"] == 0:
            return f"No price data available for '{symbol}'. Please verify the symbol is correct."
        
        # Add symbol to the response for clarity
        data["symbol"] = symbol.upper()
        
        # Format the response with readable labels
        formatted_data = {
            "symbol": symbol.upper(),
            "current_price": data["c"],
            "day_high": data["h"],
            "day_low": data["l"],
            "day_open": data["o"],
            "previous_close": data["pc"],
            "timestamp": data["t"]
        }
        
        # Return formatted JSON string
        return json.dumps(formatted_data, indent=2)
        
    except Exception as e:
        return f"Error getting stock price: {str(e)}"


@mcp.tool()
def get_basic_financials(symbol: str) -> str:
    """
    Get basic financial information for a company.

    Args:
        symbol: The stock symbol to look up (e.g., AAPL for Apple Inc.)

    Returns:
        str: Basic financial metrics in JSON format including P/E ratio, market cap, etc.
    """
    try:
        # Get basic financials from Finnhub
        data = finnhub_client.company_basic_financials(symbol.upper(), 'all')
        
        if not data or "metric" not in data:
            return f"Unable to get financial information for symbol '{symbol}'."
        
        metrics = data["metric"]
        
        # Select the most important metrics
        important_metrics = {
            "symbol": symbol.upper(),
            "company_name": data.get("series", {}).get("name", "Unknown"),
            "market_capitalization": metrics.get("marketCapitalization", None),
            "pe_ratio": metrics.get("peBasicExclExtraTTM", None),
            "pb_ratio": metrics.get("pbQuarterlyTTM", None),
            "dividend_yield": metrics.get("dividendYieldIndicatedAnnual", None),
            "52_week_high": metrics.get("52WeekHigh", None),
            "52_week_low": metrics.get("52WeekLow", None),
            "52_week_change": metrics.get("52WeekPriceReturnDaily", None),
            "beta": metrics.get("beta", None),
            "eps_ttm": metrics.get("epsBasicExclExtraItemsTTM", None),
            "revenue_per_share_ttm": metrics.get("revenuePerShareTTM", None),
            "revenue_growth_ttm": metrics.get("revenueGrowthTTM3Y", None),
            "debt_to_equity": metrics.get("totalDebtEquityQuarterly", None),
            "roa": metrics.get("roaTTM", None),
            "roe": metrics.get("roeTTM", None)
        }
        
        # Return formatted JSON string
        return json.dumps(important_metrics, indent=2)
        
    except Exception as e:
        return f"Error getting financial information: {str(e)}"


@mcp.tool()
def get_market_news(category: str = "general", min_id: int = 0) -> str:
    """
    Get the latest market news.

    Args:
        category: News category. Available values: general, forex, crypto, merger.
        min_id: Use this to get only news after this ID.

    Returns:
        str: Latest market news in JSON format with headlines, summaries, and URLs.
    """
    try:
        # Validate category
        valid_categories = ["general", "forex", "crypto", "merger"]
        if category.lower() not in valid_categories:
            return f"Invalid category. Please use one of: {', '.join(valid_categories)}"
        
        # Get market news from Finnhub
        data = finnhub_client.general_news(category.lower(), min_id=min_id)
        
        if not data or len(data) == 0:
            return f"No news available for category '{category}'."
        
        # Format the response
        formatted_data = {
            "category": category.lower(),
            "news_count": len(data),
            "articles": []
        }
        
        # Limit to 10 most recent articles to avoid overwhelming responses
        for item in data[:10]:
            formatted_data["articles"].append({
                "id": item.get("id", 0),
                "headline": item.get("headline", "No headline"),
                "summary": item.get("summary", "No summary available"),
                "source": item.get("source", "Unknown source"),
                "datetime": item.get("datetime", 0),
                "url": item.get("url", ""),
                "related_symbols": item.get("related", [])
            })
        
        # Return formatted JSON string
        return json.dumps(formatted_data, indent=2)
        
    except Exception as e:
        return f"Error getting market news: {str(e)}"


@mcp.tool()
def get_company_news(symbol: str, from_date: str, to_date: str) -> str:
    """
    Get news for a specific company over a date range.

    Args:
        symbol: The stock symbol (e.g., AAPL for Apple Inc.)
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format

    Returns:
        str: Company-specific news in JSON format with headlines, summaries, and URLs.
    """
    try:
        # Validate date format (YYYY-MM-DD)
        try:
            # Simple validation of date format
            if not (len(from_date) == 10 and len(to_date) == 10):
                return "Date format must be YYYY-MM-DD"

                # Ensure the format has dashes in the right places
            if from_date[4] != '-' or from_date[7] != '-' or to_date[4] != '-' or to_date[7] != '-':
                return "Date format must be YYYY-MM-DD (with dashes)"

        except Exception:
            return "Invalid date format. Please use YYYY-MM-DD format."

            # Get company news from Finnhub - keep the dashes in the dates
        data = finnhub_client.company_news(symbol.upper(), _from=from_date, to=to_date)

        if not data or len(data) == 0:
            return f"No news available for {symbol} between {from_date} and {to_date}."

            # Format the response
        formatted_data = {
            "symbol": symbol.upper(),
            "from_date": from_date,
            "to_date": to_date,
            "news_count": len(data),
            "articles": []
        }

        # Limit to 10 most recent articles to avoid overwhelming responses
        for item in data[:10]:
            formatted_data["articles"].append({
                "headline": item.get("headline", "No headline"),
                "summary": item.get("summary", "No summary available"),
                "source": item.get("source", "Unknown source"),
                "datetime": item.get("datetime", 0),
                "url": item.get("url", ""),
                "related_symbols": item.get("related", [])
            })

            # Return formatted JSON string
        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error getting company news: {str(e)}"


@mcp.tool()
def get_stock_candles(symbol: str, resolution: str = "D", from_time: str = None, to_time: str = None) -> str:
    """
    Get historical price data (candles) for a stock.
    Note: This is a premium feature, your API key may not have access to this feature.

    Args:
        symbol: The stock symbol (e.g., AAPL for Apple Inc.)
        resolution: Time interval between data points. Supported values: 1, 5, 15, 30, 60, D, W, M (minutes, day, week, month)
        from_time: Start time in YYYY-MM-DD format or Unix timestamp
        to_time: End time in YYYY-MM-DD format or Unix timestamp

    Returns:
        str: Historical price data in JSON format with open, high, low, close values.
    """
    try:
        # time and datetime are already imported at the top of the file

        # Handle time parameters
        if from_time is None:
            # Default to 30 days ago
            from_timestamp = int(time.time()) - (30 * 24 * 60 * 60)
        else:
            # Try to parse as YYYY-MM-DD
            try:
                if '-' in from_time:
                    dt = datetime.strptime(from_time, '%Y-%m-%d')
                    from_timestamp = int(dt.timestamp())
                else:
                    # Assume it's already a timestamp
                    from_timestamp = int(from_time)
            except ValueError:
                return "Invalid from_time format. Use YYYY-MM-DD or Unix timestamp."

        if to_time is None:
            # Default to current time
            to_timestamp = int(time.time())
        else:
            # Try to parse as YYYY-MM-DD
            try:
                if '-' in to_time:
                    dt = datetime.strptime(to_time, '%Y-%m-%d')
                    to_timestamp = int(dt.timestamp())
                else:
                    # Assume it's already a timestamp
                    to_timestamp = int(to_time)
            except ValueError:
                return "Invalid to_time format. Use YYYY-MM-DD or Unix timestamp."

        # Validate resolution
        valid_resolutions = ["1", "5", "15", "30", "60", "D", "W", "M"]
        if resolution not in valid_resolutions:
            return f"Invalid resolution. Please use one of: {', '.join(valid_resolutions)}"

        # Get stock candles from Finnhub
        data = finnhub_client.stock_candles(symbol.upper(), resolution, from_timestamp, to_timestamp)

        if not data or data.get("s") != "ok":
            return f"Unable to get candle data for {symbol}. Status: {data.get('s', 'unknown')}"

        # Format the response
        formatted_data = {
            "symbol": symbol.upper(),
            "resolution": resolution,
            "from_time": from_timestamp,
            "to_time": to_timestamp,
            "status": data.get("s"),
            "candle_count": len(data.get("t", [])),
            "candles": []
        }

        # Combine the data into candles
        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        for i in range(min(len(timestamps), 100)):  # Limit to 100 candles max
            formatted_data["candles"].append({
                "timestamp": timestamps[i],
                "datetime": datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S'),
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": closes[i] if i < len(closes) else None,
                "volume": volumes[i] if i < len(volumes) else None
            })

        # Return formatted JSON string
        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error getting stock candles: {str(e)}"

@mcp.tool()
def scrape_article(url: str, timeout_s: int = 15) -> str:
    """
    Scrape a news article URL and return extracted text content.

    Args:
        url: The article URL to scrape (http/https).
        timeout_s: Request timeout in seconds.

    Returns:
        str: JSON payload with title and extracted text.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Invalid URL scheme. Only http/https are supported."

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return "URL did not return HTML content."

        title, text = _extract_readable_text(response.text)

        payload = {
            "url": url,
            "title": title,
            "text": text,
            "content_length": len(text),
            "truncated": len(text) >= MAX_ARTICLE_CHARS,
        }

        return json.dumps(payload, indent=2)

    except httpx.HTTPError as e:
        return f"HTTP error scraping article: {str(e)}"
    except Exception as e:
        return f"Error scraping article: {str(e)}"

@mcp.prompt("stock_analysis")
def stock_analysis_prompt():
    """
    Analyze a stock by looking up its basic information, current price, and financial metrics.

    Example usage:
    I want to analyze Apple stock. First, look up the symbol for Apple.
    Then get the current price and basic financial information.
    Finally, summarize whether this looks like a good investment based on P/E ratio,
    dividend yield, and recent price movements.
    """
    return """                                                                                                                                             
    I need to analyze a stock for potential investment. Please help me with the following:                                                                     
    
    1. Look up the symbol for {company_name}                                                                                                                   
    2. Get the current price for the best matching symbol                                                                                                      
    3. Retrieve the basic financial information                                                                                                                
    4. Based on the P/E ratio, dividend yield, and recent price movements, provide a brief assessment of whether this might be a good investment opportunity   
    
    Please format your analysis in a clear, structured way with sections for each piece of information.                                                        
    """

@mcp.prompt("market_overview")
def market_overview_prompt():
    """
    Get a comprehensive market overview including general market news and data for major indices.

    Example usage:
    Give me a market overview with the latest general news and current prices for major indices like
    AAPL, MSFT, GOOGL, AMZN, and SPY.
    """
    return """                                                                                                                                             
    Please provide a comprehensive market overview with:                                                                                                       
    
    1. The latest general market news (use the get_market_news tool with category="general")                                                                   
    2. Current prices for these major stocks and indices:                                                                                                      
       - Apple (AAPL)                                                                                                                                          
       - Microsoft (MSFT)                                                                                                                                      
       - S&P 500 ETF (SPY)                                                                                                                                     
       - {additional_symbol_1}                                                                                                                                 
       - {additional_symbol_2}                                                                                                                                 
    
    3. Summarize the overall market sentiment based on the news and price movements                                                                            
    """

@mcp.prompt("stock_price_history")
def stock_price_history_prompt():
    """
    Analyze the historical price movement of a stock over a specific time period.

    Example usage:
    Show me the price history for TSLA over the past month with daily candles.
    Identify key support and resistance levels.
    """
    return """                                                                                                                                             
    I'd like to analyze the historical price movement for {symbol} from {start_date} to {end_date}.                                                            
    
    Please:                                                                                                                                                    
    1. Get the stock candles with a {resolution} resolution (use "D" for daily, "W" for weekly)                                                                
    2. Identify the highest and lowest prices during this period                                                                                               
    3. Calculate the overall percentage change                                                                                                                 
    4. Note any significant price movements or patterns                                                                                                        
    5. If possible, identify potential support and resistance levels based on the price history                                                                
    """

@mcp.prompt("company_news_analysis")
def company_news_analysis_prompt():
    """
    Analyze recent news for a specific company and its potential impact on stock price.

    Example usage:
    Analyze recent news for NVDA from 2023-01-01 to 2023-01-31 and tell me how it might
    affect the stock price.
    """
    return """                                                                                                                                             
    Please analyze recent news for {symbol} from {start_date} to {end_date} and assess its potential impact on the stock price:                                
    
    1. Retrieve company-specific news for the specified period                                                                                                 
    2. Get the current stock price for context                                                                                                                 
    3. Summarize the key news items, focusing on:                                                                                                              
       - Product announcements                                                                                                                                 
       - Financial results                                                                                                                                     
       - Management changes                                                                                                                                    
       - Regulatory issues                                                                                                                                     
       - Competitive developments                                                                                                                              
    4. Provide an assessment of whether these news items are likely to have a positive, negative, or neutral impact on the stock price                         
    """

if __name__ == "__main__":
    # Initialize and run the server                                                                                                                        
    mcp.run(transport='stdio')        
