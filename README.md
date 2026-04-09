# Stock Market MCP Server

A Machine Conversation Protocol (MCP) server that provides stock market data and analysis tools using the Finnhub API.

## Features

- Stock symbol lookup
- Real-time stock prices
- Basic financial metrics
- Market and company news
- Historical price data (candles)
- Pre-built prompts for common analysis tasks

## Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- Finnhub API key (get one for free at [finnhub.io](https://finnhub.io/))
- Claude Desktop (for using the MCP server with Claude)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-market-mcp-server.git
cd stock-market-mcp-server
```

### 2. Set Up Environment with uv

uv is a fast, reliable Python package installer and virtual environment manager. Here's how to set up your environment:

```bash
# Install uv if you don't have it already
pip install uv

# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
uv add -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env.local` file in the project root with your Finnhub API key:

```
FINNHUB_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Finnhub API key.

## Running the MCP Server

To start the MCP server:

```bash
uv run stock-market-server.py
```

The server runs in stdio mode by default, which means it reads from standard input and writes to standard output.

## Using with Claude Desktop

To use this MCP server with Claude Desktop:

1. Start Claude Desktop
2. Go to Settings > Developer
3. Enable "Developer Mode"
4. Click "Add Tool"
5. Select "Local MCP Server"
6. Configure the tool:
   - Name: Stock Market
   - Command: The full path to your Python executable and the script
     - Example: `/Users/username/.venv/bin/python /Users/username/stock-market-mcp-server/stock-market-server.py`
   - Working Directory: The full path to your project directory
7. Click "Save"

Now you can use the stock market tools in your conversations with Claude!

Alternatively, you can add the tool to your Claude Desktop tools list manually. 
You can find the configuration file in the installation directory of Claude Desktop. On a MacBook Pro, 
it's located at ` ~/Library/Application\ Support/Claude/claude_desktop_config.json`.
Add the following JSON to the `tools` array, note modify the path to suite your environment:
```json
"stock-market": {
   "command": "uv",
   "args": [
     "--directory",
     "<YOUR_PATH_GOES_HERE>/stock-market-mcp-server",
     "run",
     "stock_market_server.py"
   ]
 }
```
## Using with Stock Market Client

To use the MCP server with the Stock Market Client, follow these steps:

1. You will need an Anthropic API key. You can get one for free at [Anthropic](https://console.anthropic.com/signup).
2. Update the _.env_ file with your API key. ```ANTHROPIC_API_KEY=your_api_key_here```
3. Start the MCP Client with the following command: ```uv run stock_market_client.py ```

## News Agent (Top 5 Relevant News)

This project now includes a focused news agent that:
- Pulls company or market news from the MCP server
- Scrapes the original article URLs
- Ranks relevance to the user's query
- Summarizes the top 5 most relevant items, tailored to the query

Run it with:

```bash
uv run news_agent_client.py
```

## LangGraph + LangChain News Chat

This version uses LangGraph to orchestrate planning, tool calls, scraping, summarization,
and question answering. It returns structured JSON with an answer plus top 5 news items.

Run it with:

```bash
uv run langgraph_news_agent.py
```


## Available Tools

The MCP server provides the following tools:

- `get_stock_symbol_lookup`: Search for stock symbols by company name
- `get_stock_price`: Get the latest price for a stock
- `get_basic_financials`: Get key financial metrics for a company
- `get_market_news`: Get the latest market news by category
- `get_company_news`: Get news for a specific company over a date range
- `get_stock_candles`: Get historical price data for a stock
- `scrape_article`: Scrape a news article URL and extract the main text

## Available Prompts

The server includes these pre-built prompts:

- `stock_analysis`: Analyze a stock for potential investment
- `market_overview`: Get a comprehensive market overview
- `stock_price_history`: Analyze historical price movements
- `company_news_analysis`: Analyze news and its impact on stock price

## Example Usage Prompts

Here are some examples of how to use the tools with Claude:

1. **Looking up a stock symbol**:
   "Can you look up the stock symbol for Apple?"

2. **Getting current stock price**:
   "What's the current price of AAPL?"

3. **Using a prompt**:
   "I'd like to analyze Tesla stock as a potential investment."

4. **Combining multiple tools**:
   "Give me a market overview with the latest news and the current prices for AAPL, MSFT, and TSLA."

## Troubleshooting

- **API Key Issues**: Make sure your Finnhub API key is correctly set in the `.env.local` file
- **Module Not Found Errors**: Ensure you've activated the virtual environment and installed all dependencies
- **Connection Issues**: Check your internet connection and Finnhub API status

## License

[MIT License](LICENSE)
