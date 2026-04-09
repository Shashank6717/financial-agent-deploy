import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# -----------------------------
# Models
# -----------------------------

class ArticleSummary(BaseModel):
    headline: str = Field(..., description="Article headline")
    source: str = Field(..., description="Publisher/source name")
    date: str = Field(..., description="YYYY-MM-DD date")
    link: str = Field(..., description="Source URL or N/A")
    summary: str = Field(..., description="1-3 sentence summary tailored to the query")
    relevance: str = Field(..., description="Short clause explaining relevance to the query")

class NewsSummaryResponse(BaseModel):
    query: str
    top_articles: list[ArticleSummary]

class ToolCall(BaseModel):
    name: str = Field(..., description="MCP tool name to call")
    args: dict = Field(default_factory=dict, description="Arguments for the tool")

class PlanResponse(BaseModel):
    actions: list[ToolCall] = Field(default_factory=list, min_length=1, description="Tool calls to execute in order")
    user_question: Optional[str] = Field(default=None, description="Question to ask user if clarification is needed")
    rationale: Optional[str] = Field(default=None, description="Short reason for the plan")

@dataclass
class Article:
    headline: str
    summary: str
    source: str
    datetime_ts: int
    url: str
    related_symbols: list
    scraped_title: str = ""
    scraped_text: str = ""

# -----------------------------
# LLM Wrapper
# -----------------------------

class LLMClient:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if self.gemini_api_key:
            self.provider = "gemini"
            self.client = genai.Client(api_key=self.gemini_api_key)
        elif self.anthropic_api_key:
            from anthropic import Anthropic
            self.provider = "anthropic"
            self.client = Anthropic(api_key=self.anthropic_api_key)
        else:
            raise ValueError("No API key found. Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")

    def generate_json(self, prompt: str, schema: Optional[type[BaseModel]] = None, max_tokens: int = 1024) -> str:
        if self.provider == "gemini":
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
            if schema:
                config.response_schema = schema

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            return response.text or ""

        # Anthropic fallback
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_text(self, prompt: str, schema: Optional[type[BaseModel]] = None, max_tokens: int = 2048) -> str:
        if self.provider == "gemini":
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if schema else "text/plain",
            )
            if schema:
                config.response_schema = schema

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            return response.text or ""

        # Anthropic fallback
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

# -----------------------------
# MCP Tooling
# -----------------------------

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None

    async def connect(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

    async def close(self):
        await self.exit_stack.aclose()

    async def list_tools(self) -> list[dict]:
        response = await self.session.list_tools()
        tools = []
        for tool in response.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            })
        return tools

    async def call_tool_text(self, tool_name: str, tool_args: dict) -> str:
        result = await self.session.call_tool(tool_name, tool_args)
        if hasattr(result, "content"):
            for item in result.content:
                if getattr(item, "type", "") == "text":
                    return item.text
        return str(result)

# -----------------------------
# Planning + Summarization
# -----------------------------

class Planner:
    def __init__(self, llm: LLMClient, mcp: MCPClient):
        self.llm = llm
        self.mcp = mcp

    async def plan(self, query: str) -> PlanResponse:
        tools = await self.mcp.list_tools()
        schema = PlanResponse.model_json_schema()
        prompt = f"""User query: {query}

You are a planner for a financial news agent. Decide which MCP tools to call and with what arguments.
You may ask a clarifying question if needed. Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Available tools:
{json.dumps(tools, indent=2)}

Rules:
- Use the tool names exactly as listed.
- If you can answer with available tools, do not ask a question.
- Prefer fewer tool calls, but make sure results will be relevant.
- If the query mentions a company but no symbol, use get_stock_symbol_lookup first.
- For company news, pick a reasonable recent date range if none is specified.
- For general financial news, use get_market_news with the most relevant category.
"""

        raw = self.llm.generate_json(prompt, max_tokens=1024)
        payload = _parse_json_with_repair(self.llm, raw, schema, purpose="tool plan")
        if payload is None:
            return PlanResponse(actions=[])
        return PlanResponse.model_validate(payload)

class Summarizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize(self, query: str, articles: list[Article]) -> str:
        if not articles:
            return "I couldn't find any relevant news items for that query."

        def format_dt(ts: int) -> str:
            if not ts:
                return "Unknown date"
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

        items = []
        for idx, article in enumerate(articles, start=1):
            items.append(
                f"""Article {idx}:
Headline: {article.headline}
Source: {article.source}
Date: {format_dt(article.datetime_ts)}
URL: {article.url}
Summary snippet: {truncate(article.summary, 600)}
Scraped title: {truncate(article.scraped_title, 300)}
Scraped text: {truncate(article.scraped_text, 4000)}
"""
            )

        schema = NewsSummaryResponse.model_json_schema()
        prompt = f"""User query: {query}

You are a financial news assistant. Summarize the top 5 most relevant news items for the query using the articles provided below.
Return ONLY valid JSON that matches the following schema exactly. Do NOT include any markdown formatting, preamble, or postamble.
{json.dumps(schema, indent=2)}

Rules:
1. The 'top_articles' list must have exactly the most relevant items (up to 5).
2. The 'link' field must be a valid URL or 'N/A' if missing.
3. The 'summary' must be a concise 1-3 sentence summary tailored to the user query.
4. The 'relevance' must be a short clause explaining why the article was included.
5. Date format: YYYY-MM-DD.

Articles:
{chr(10).join(items)}
"""

        raw = self.llm.generate_text(prompt, max_tokens=2048)
        payload = _parse_json_with_repair(self.llm, raw, schema, purpose="news summary")
        if payload is None:
            # Fallback error response as JSON string
            return json.dumps({"error": "Failed to parse structured JSON from the LLM.", "query": query}, indent=2)

        try:
            validated = NewsSummaryResponse.model_validate(payload)
            # Ensure we only return up to top 5
            validated.top_articles = validated.top_articles[:5]
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            return json.dumps({"error": f"Validation failed: {str(e)}", "query": query}, indent=2)

# -----------------------------
# Agent
# -----------------------------

class NewsAgent:
    def __init__(self, llm: LLMClient, mcp: MCPClient):
        self.llm = llm
        self.mcp = mcp
        self.planner = Planner(llm, mcp)
        self.summarizer = Summarizer(llm)

    async def run(self, query: str) -> str:
        plan = await self.planner.plan(query)
        if plan.user_question and not plan.actions:
            return f"CLARIFY: {plan.user_question}"
        if not plan.actions:
            return "CLARIFY: I couldn't determine which tools to call. Please specify a company, symbol, or market category."

        all_articles: list[Article] = []
        for action in plan.actions:
            tool_text = await self.mcp.call_tool_text(action.name, action.args)
            all_articles.extend(_collect_articles_from_tool(action.name, tool_text))

        all_articles = dedupe_articles(all_articles)
        all_articles = await _scrape_articles(self.mcp, all_articles)

        # Ensure at least 5 by falling back to general market news if needed
        if len(all_articles) < 5:
            fallback_text = await self.mcp.call_tool_text("get_market_news", {"category": "general"})
            all_articles.extend(_collect_articles_from_tool("get_market_news", fallback_text))
            all_articles = dedupe_articles(all_articles)
            all_articles = await _scrape_articles(self.mcp, all_articles)

        return self.summarizer.summarize(query, all_articles[:10])

# -----------------------------
# Helpers
# -----------------------------

def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

def _parse_json_with_repair(llm: LLMClient, raw: str, schema: dict, purpose: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_block(raw)
        if extracted:
            return json.loads(extracted)

    repair_prompt = f"""The following output is not valid JSON for the required schema.
Purpose: {purpose}
Return ONLY valid JSON that matches this schema exactly (no extra text).

Schema:
{json.dumps(schema, indent=2)}

Broken output:
{raw}
"""

    fixed = llm.generate_json(repair_prompt, max_tokens=1024)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        extracted = extract_json_block(fixed)
        if extracted:
            return json.loads(extracted)
    return None

def dedupe_articles(articles: list[Article]) -> list[Article]:
    seen = set()
    deduped = []
    for article in articles:
        key = (article.url or article.headline, article.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped

async def _scrape_articles(mcp: MCPClient, articles: list[Article]) -> list[Article]:
    for article in articles:
        if not article.url:
            continue
        scrape_text = await mcp.call_tool_text("scrape_article", {"url": article.url})
        try:
            data = json.loads(scrape_text)
        except json.JSONDecodeError:
            continue
        article.scraped_title = data.get("title", "")
        article.scraped_text = data.get("text", "")
    return articles

def _collect_articles_from_tool(tool_name: str, tool_text: str) -> list[Article]:
    articles: list[Article] = []
    try:
        data = json.loads(tool_text)
    except json.JSONDecodeError:
        return articles

    if tool_name in ("get_company_news", "get_market_news"):
        for item in data.get("articles", []):
            articles.append(Article(
                headline=item.get("headline", "No headline"),
                summary=item.get("summary", ""),
                source=item.get("source", "Unknown"),
                datetime_ts=item.get("datetime", 0),
                url=item.get("url", ""),
                related_symbols=item.get("related_symbols", []),
            ))
    return articles

# -----------------------------
# CLI
# -----------------------------

class NewsAgentClient:
    def __init__(self):
        self.mcp = MCPClient()
        self.llm = LLMClient()
        self.agent = NewsAgent(self.llm, self.mcp)

    async def connect_to_server(self, server_script_path: str):
        await self.mcp.connect(server_script_path)

    async def cleanup(self):
        await self.mcp.close()

    async def process_query(self, query: str) -> str:
        return await self.agent.run(query)

    async def chat_loop(self):
        engine = "Gemini" if self.llm.provider == "gemini" else "Anthropic"
        print(f"\nNews Agent Started (Powered by {engine})!")
        print("Type your query or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                if not query:
                    continue

                response = await self.process_query(query)
                if response.startswith("CLARIFY: "):
                    follow_up = response.replace("CLARIFY: ", "").strip()
                    print("\n" + follow_up)
                    user_answer = input("\nAnswer: ").strip()
                    if user_answer:
                        combined = f"{query}\nUser clarification: {user_answer}"
                        response = await self.process_query(combined)
                        print("\n" + response)
                    else:
                        print("\nNo answer provided. Please ask again with more detail.")
                else:
                    print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    client = NewsAgentClient()
    try:
        await client.connect_to_server("./stock_market_server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
