import json
from typing import Optional, Any
from pydantic import BaseModel, Field, create_model

SYSTEM_PROMPT = """You are a financial analysis assistant with access to real-time market data and news.

## How you work
1. Receive a user query
2. Fetch relevant data using tools (prices, news, articles, financials)
3. READ and REASON over what the tools return
4. Answer the query in your own words — synthesized, clear, and direct

## Output rules
- NEVER dump raw tool output or JSON at the user. Always synthesize it into a proper answer.
- For news/article queries: fetch the articles, read their content, then write a summary that directly answers the question. Cite sources by headline or publication — not raw URLs.
- For price/financial queries: present numbers cleanly (e.g. "$182.34", "P/E: 28.4x") with a brief interpretation.
- Use bullet points or short paragraphs — never walls of raw data.
- Always ground your answer in what the tools returned (if used). Never hallucinate facts.
- If it's a general question or greeting (e.g. "hi"), answer it nicely and naturally using your own knowledge.

## News/article workflow
When a user asks something that requires news or article content:
1. Call get_market_news or get_company_news to get article list
2. Call scrape_article on the most relevant articles (at least 5 — always aim for 5 or more)
3. Synthesize the scraped content into a direct answer to the query
4. Format your response like this:

**[Your synthesized answer here — 2–4 sentences directly answering the query]**

📰 Sources:
- Headline 1 — brief 1-line takeaway
- Headline 2 — brief 1-line takeaway
- Headline 3 — brief 1-line takeaway
- Headline 4 — brief 1-line takeaway
- Headline 5 — brief 1-line takeaway
(include more if available)

## Other rules
- For ambiguous tickers (e.g. "Apple"), resolve to symbol (AAPL) first
- If a tool fails, try an alternative — don't give up after one error
"""

_JSON_TO_PYTHON: dict[str, type] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}

def _build_args_schema(model_name: str, json_schema: dict) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    properties: dict = json_schema.get("properties", {})
    required_set: set[str] = set(json_schema.get("required", []))

    for field_name, field_schema in properties.items():
        raw_type = field_schema.get("type", "string")
        python_type: type = _JSON_TO_PYTHON.get(raw_type, str)
        description: str = field_schema.get("description", "")

        if field_name in required_set:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            fields[field_name] = (Optional[python_type], Field(None, description=description))

    return create_model(model_name, **fields)

def _format_tool_result(tool_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()

    if tool_name in ("get_market_news", "get_company_news"):
        if not isinstance(data, list):
            return raw

        lines = [f"Found {len(data)} articles:\n"]
        for i, article in enumerate(data[:10], 1):          # cap at 10
            headline  = article.get("headline") or article.get("title", "No title")
            source    = article.get("source", "")
            url       = article.get("url", "")
            summary   = article.get("summary", "")
            timestamp = article.get("datetime", "")

            lines.append(f"{i}. [{headline}]")
            if source:    lines.append(f"   Source: {source}")
            if timestamp: lines.append(f"   Time:   {timestamp}")
            if summary:   lines.append(f"   Summary: {summary}")
            if url:       lines.append(f"   URL: {url}")
            lines.append("")

        return "\n".join(lines)

    if tool_name == "scrape_article":
        if isinstance(data, dict):
            title   = data.get("title", "")
            content = data.get("content") or data.get("text") or data.get("body", "")
            source  = data.get("source", "")
            parts = []
            if title:   parts.append(f"Title: {title}")
            if source:  parts.append(f"Source: {source}")
            if content: parts.append(f"\n{content.strip()}")
            return "\n".join(parts) if parts else raw
        return str(data)

    if tool_name == "get_stock_price":
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)

    if tool_name == "get_basic_financials":
        if isinstance(data, dict):
            metric = data.get("metric", data)
            lines = []
            for k, v in (metric.items() if isinstance(metric, dict) else {}.items()):
                if v is not None:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines) if lines else raw

    if tool_name == "get_stock_candles":
        if isinstance(data, dict):
            status = data.get("s", "")
            closes = data.get("c", [])
            opens  = data.get("o", [])
            highs  = data.get("h", [])
            lows   = data.get("l", [])
            times  = data.get("t", [])
            if status == "no_data":
                return "No candle data available for this period."
            lines = [f"Status: {status}", f"Candles returned: {len(closes)}"]
            if closes:
                lines.append(f"Latest close:  {closes[-1]}")
                lines.append(f"Latest open:   {opens[-1] if opens else 'N/A'}")
                lines.append(f"Latest high:   {highs[-1] if highs else 'N/A'}")
                lines.append(f"Latest low:    {lows[-1] if lows else 'N/A'}")
                lines.append(f"Price range (all): {min(lows or closes):.2f} – {max(highs or closes):.2f}")
            return "\n".join(lines)

    return json.dumps(data, indent=2)
