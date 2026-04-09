import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, TypedDict, List
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

load_dotenv()

# -----------------------------
# Pydantic Models
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

class FinalAnswerResponse(BaseModel):
    query: str
    answer: str
    top_articles: list[ArticleSummary]

class ToolCall(BaseModel):
    name: str = Field(..., description="MCP tool name to call")
    args: dict = Field(default_factory=dict, description="Arguments for the tool")

class PlanResponse(BaseModel):
    actions: list[ToolCall] = Field(default_factory=list, min_length=1, description="Tool calls to execute in order")
    user_question: Optional[str] = Field(default=None, description="Question to ask user if clarification is needed")
    rationale: Optional[str] = Field(default=None, description="Short reason for the plan")

# -----------------------------
# Article Model
# -----------------------------

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
            self.llm_plan = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                google_api_key=self.gemini_api_key,
            )
            self.llm_summarize = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=self.gemini_api_key,
            )
        elif self.anthropic_api_key:
            self.provider = "anthropic"
            self.llm_plan = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.0,
                api_key=self.anthropic_api_key,
            )
            self.llm_summarize = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.2,
                api_key=self.anthropic_api_key,
            )
        else:
            raise ValueError("No API key found. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")

    def _invoke(self, llm, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        if isinstance(response.content, str):
            return response.content
        return json.dumps(response.content)

    def generate_json(self, prompt: str) -> str:
        return self._invoke(self.llm_plan, prompt)

    def generate_text(self, prompt: str) -> str:
        return self._invoke(self.llm_summarize, prompt)

# -----------------------------
# MCP Client
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
            env=None,
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
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )
        return tools

    async def call_tool_text(self, tool_name: str, tool_args: dict) -> str:
        result = await self.session.call_tool(tool_name, tool_args)
        if hasattr(result, "content"):
            for item in result.content:
                if getattr(item, "type", "") == "text":
                    return item.text
        return str(result)

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

def parse_json_with_repair(llm: LLMClient, raw: str, schema: dict, purpose: str) -> Optional[dict]:
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
    fixed = llm.generate_json(repair_prompt)
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

def collect_articles_from_tool(tool_name: str, tool_text: str) -> list[Article]:
    articles: list[Article] = []
    try:
        data = json.loads(tool_text)
    except json.JSONDecodeError:
        return articles

    if tool_name in ("get_company_news", "get_market_news"):
        for item in data.get("articles", []):
            articles.append(
                Article(
                    headline=item.get("headline", "No headline"),
                    summary=item.get("summary", ""),
                    source=item.get("source", "Unknown"),
                    datetime_ts=item.get("datetime", 0),
                    url=item.get("url", ""),
                    related_symbols=item.get("related_symbols", []),
                )
            )
    return articles

async def scrape_articles(mcp: MCPClient, articles: list[Article]) -> list[Article]:
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

# -----------------------------
# LangGraph State
# -----------------------------

class NewsState(TypedDict):
    query: str
    plan: Optional[PlanResponse]
    articles: List[Article]
    summary: Optional[NewsSummaryResponse]
    answer: Optional[str]

# -----------------------------
# Nodes
# -----------------------------

async def plan_node(state: NewsState, llm: LLMClient, mcp: MCPClient) -> NewsState:
    tools = await mcp.list_tools()
    schema = PlanResponse.model_json_schema()
    prompt = f"""User query: {state['query']}

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
    raw = llm.generate_json(prompt)
    payload = parse_json_with_repair(llm, raw, schema, purpose="tool plan")
    if payload is None:
        state["plan"] = PlanResponse(actions=[])
        return state
    try:
        state["plan"] = PlanResponse.model_validate(payload)
    except ValidationError:
        state["plan"] = PlanResponse(actions=[])
    return state

async def tools_node(state: NewsState, mcp: MCPClient) -> NewsState:
    plan = state.get("plan")
    articles: list[Article] = []

    if not plan or not plan.actions:
        state["articles"] = []
        return state

    for action in plan.actions:
        tool_text = await mcp.call_tool_text(action.name, action.args)
        articles.extend(collect_articles_from_tool(action.name, tool_text))

    state["articles"] = dedupe_articles(articles)
    return state

async def ensure_min_articles_node(state: NewsState, mcp: MCPClient) -> NewsState:
    if len(state.get("articles", [])) >= 5:
        return state

    fallback_text = await mcp.call_tool_text("get_market_news", {"category": "general"})
    fallback_articles = collect_articles_from_tool("get_market_news", fallback_text)
    combined = dedupe_articles(state.get("articles", []) + fallback_articles)
    state["articles"] = combined
    return state

async def scrape_node(state: NewsState, mcp: MCPClient) -> NewsState:
    articles = state.get("articles", [])
    state["articles"] = await scrape_articles(mcp, articles)
    return state

async def summarize_node(state: NewsState, llm: LLMClient) -> NewsState:
    articles = state.get("articles", [])
    if not articles:
        state["summary"] = None
        return state

    def format_dt(ts: int) -> str:
        if not ts:
            return "Unknown date"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    items = []
    for idx, article in enumerate(articles[:10], start=1):
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
    prompt = f"""User query: {state['query']}

You are a financial news assistant. Summarize the top 5 most relevant news items for the query using the articles below.
Return ONLY valid JSON that matches this schema exactly (no markdown, no extra text):
{json.dumps(schema, indent=2)}

Rules:
- Always include a "link" field and use "N/A" if missing.
- Keep tone neutral and factual.
- If an article lacks detail, say so briefly in "summary".
- Use date format YYYY-MM-DD.

Articles:
{chr(10).join(items)}
"""
    raw = llm.generate_text(prompt)
    payload = parse_json_with_repair(llm, raw, schema, purpose="news summary")
    if payload is None:
        state["summary"] = None
        return state

    try:
        state["summary"] = NewsSummaryResponse.model_validate(payload)
    except ValidationError:
        state["summary"] = None
    return state

async def answer_node(state: NewsState, llm: LLMClient) -> NewsState:
    summary = state.get("summary")
    if not summary:
        state["answer"] = "I couldn't find reliable news to answer that. Please try a more specific query."
        return state

    summary_json = summary.model_dump_json(indent=2)
    prompt = f"""You are a financial Q&A assistant. Use the news summary below to answer the user's question.
If the news does not fully answer the question, say so clearly.

User question: {state['query']}

News summary:
{summary_json}

Write a concise, factual answer grounded in the news above.
"""
    state["answer"] = llm.generate_text(prompt).strip()
    return state

# -----------------------------
# Graph Builder
# -----------------------------

def build_graph(llm: LLMClient, mcp: MCPClient):
    graph = StateGraph(NewsState)

    graph.add_node("plan", lambda state: plan_node(state, llm, mcp))
    graph.add_node("tools", lambda state: tools_node(state, mcp))
    graph.add_node("ensure_min", lambda state: ensure_min_articles_node(state, mcp))
    graph.add_node("scrape", lambda state: scrape_node(state, mcp))
    graph.add_node("summarize", lambda state: summarize_node(state, llm))
    graph.add_node("answer", lambda state: answer_node(state, llm))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "tools")
    graph.add_edge("tools", "ensure_min")
    graph.add_edge("ensure_min", "scrape")
    graph.add_edge("scrape", "summarize")
    graph.add_edge("summarize", "answer")
    graph.add_edge("answer", END)

    return graph.compile()

# -----------------------------
# CLI
# -----------------------------

class LangGraphNewsAgentClient:
    def __init__(self):
        self.mcp = MCPClient()
        self.llm = LLMClient()
        self.app = build_graph(self.llm, self.mcp)

    async def connect_to_server(self, server_script_path: str):
        await self.mcp.connect(server_script_path)

    async def cleanup(self):
        await self.mcp.close()

    async def process_query(self, query: str) -> str:
        state: NewsState = {
            "query": query,
            "plan": None,
            "articles": [],
            "summary": None,
            "answer": None,
        }
        result = await self.app.ainvoke(state)

        summary = result.get("summary")
        answer = result.get("answer", "")
        if not summary:
            return "I couldn't find any relevant news items for that query."

        final = FinalAnswerResponse(
            query=query,
            answer=answer or "",
            top_articles=summary.top_articles[:5],
        )
        return final.model_dump_json(indent=2)

    async def chat_loop(self):
        engine = "Gemini" if self.llm.provider == "gemini" else "Anthropic"
        print(f"\nLangGraph News Agent Started (Powered by {engine})!")
        print("Type your query or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                if not query:
                    continue
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    client = LangGraphNewsAgentClient()
    try:
        await client.connect_to_server("./stock_market_server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
