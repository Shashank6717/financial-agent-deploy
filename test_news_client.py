# """
# Financial Agent — LangGraph + Gemini 2.5 Flash + MCP
# =====================================================
# Architecture:
#   START → [agent node] ──(has tool calls?)──► [tools node] → [agent node] → ...
#                          └──(no tool calls)──► END

# The agent maintains full conversation history across turns for context.
# """

# import asyncio
# from typing import Optional, Any
# from contextlib import AsyncExitStack

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.tools import StructuredTool
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from pydantic import BaseModel, Field, create_model
# from dotenv import load_dotenv

# load_dotenv()

# # ─────────────────────────────────────────────
# # SYSTEM PROMPT
# # ─────────────────────────────────────────────

# SYSTEM_PROMPT = """You are a financial analysis assistant with access to real-time market data and news.

# ## How you work
# 1. Receive a user query
# 2. Fetch relevant data using tools (prices, news, articles, financials)
# 3. READ and REASON over what the tools return
# 4. Answer the query in your own words — synthesized, clear, and direct

# ## Output rules
# - NEVER dump raw tool output or JSON at the user. Always synthesize it into a proper answer.
# - For news/article queries: fetch the articles, read their content, then write a summary that directly answers the question. Cite sources by headline or publication — not raw URLs.
# - For price/financial queries: present numbers cleanly (e.g. "$182.34", "P/E: 28.4x") with a brief interpretation.
# - Use bullet points or short paragraphs — never walls of raw data.
# - Always ground your answer in what the tools returned. Never hallucinate numbers or events.

# ## News/article workflow
# When a user asks something that requires news or article content:
# 1. Call get_market_news or get_company_news to get article list
# 2. Call scrape_article on the most relevant articles (top 2–3)
# 3. Synthesize the scraped content into a direct answer to the query
# 4. Format your response like this:

# **[Your synthesized answer here — 2–4 sentences directly answering the query]**

# 📰 Sources:
# - Headline 1 — brief 1-line takeaway
# - Headline 2 — brief 1-line takeaway

# ## Other rules
# - For ambiguous tickers (e.g. "Apple"), resolve to symbol (AAPL) first
# - If a tool fails, try an alternative — don't give up after one error
# """

# # ─────────────────────────────────────────────
# # SCHEMA UTILS
# # ─────────────────────────────────────────────

# _JSON_TO_PYTHON: dict[str, type] = {
#     "string":  str,
#     "integer": int,
#     "number":  float,
#     "boolean": bool,
#     "array":   list,
#     "object":  dict,
# }


# def _build_args_schema(model_name: str, json_schema: dict) -> type[BaseModel]:
#     """
#     Dynamically create a Pydantic v2 model from a JSON Schema dict.
#     Only handles flat / one-level-deep schemas — sufficient for most MCP tools.
#     """
#     fields: dict[str, Any] = {}
#     properties: dict = json_schema.get("properties", {})
#     required_set: set[str] = set(json_schema.get("required", []))

#     for field_name, field_schema in properties.items():
#         raw_type = field_schema.get("type", "string")
#         python_type: type = _JSON_TO_PYTHON.get(raw_type, str)
#         description: str = field_schema.get("description", "")

#         if field_name in required_set:
#             fields[field_name] = (python_type, Field(..., description=description))
#         else:
#             fields[field_name] = (Optional[python_type], Field(None, description=description))

#     return create_model(model_name, **fields)


# # ─────────────────────────────────────────────
# # TOOL RESULT FORMATTER
# # ─────────────────────────────────────────────

# def _format_tool_result(tool_name: str, raw: str) -> str:
#     """
#     Parse tool output (often JSON) and reformat it into clean readable text
#     so the LLM can reason over it instead of getting raw dumps.
#     """
#     import json

#     try:
#         data = json.loads(raw)
#     except (json.JSONDecodeError, TypeError):
#         # Not JSON — return as-is (e.g. scraped article text)
#         return raw.strip()

#     # ── News / article lists ──────────────────
#     if tool_name in ("get_market_news", "get_company_news"):
#         if not isinstance(data, list):
#             return raw

#         lines = [f"Found {len(data)} articles:\n"]
#         for i, article in enumerate(data[:10], 1):          # cap at 10
#             headline  = article.get("headline") or article.get("title", "No title")
#             source    = article.get("source", "")
#             url       = article.get("url", "")
#             summary   = article.get("summary", "")
#             timestamp = article.get("datetime", "")

#             lines.append(f"{i}. [{headline}]")
#             if source:    lines.append(f"   Source: {source}")
#             if timestamp: lines.append(f"   Time:   {timestamp}")
#             if summary:   lines.append(f"   Summary: {summary}")
#             if url:       lines.append(f"   URL: {url}")
#             lines.append("")

#         return "\n".join(lines)

#     # ── Scraped article content ───────────────
#     if tool_name == "scrape_article":
#         if isinstance(data, dict):
#             title   = data.get("title", "")
#             content = data.get("content") or data.get("text") or data.get("body", "")
#             source  = data.get("source", "")
#             parts = []
#             if title:   parts.append(f"Title: {title}")
#             if source:  parts.append(f"Source: {source}")
#             if content: parts.append(f"\n{content.strip()}")
#             return "\n".join(parts) if parts else raw
#         # Sometimes scrape returns plain text
#         return str(data)

#     # ── Stock price ───────────────────────────
#     if tool_name == "get_stock_price":
#         if isinstance(data, dict):
#             lines = []
#             for k, v in data.items():
#                 lines.append(f"{k}: {v}")
#             return "\n".join(lines)

#     # ── Basic financials ──────────────────────
#     if tool_name == "get_basic_financials":
#         if isinstance(data, dict):
#             metric = data.get("metric", data)
#             lines = []
#             for k, v in (metric.items() if isinstance(metric, dict) else {}.items()):
#                 if v is not None:
#                     lines.append(f"{k}: {v}")
#             return "\n".join(lines) if lines else raw

#     # ── Candles (OHLCV) ──────────────────────
#     if tool_name == "get_stock_candles":
#         if isinstance(data, dict):
#             status = data.get("s", "")
#             closes = data.get("c", [])
#             opens  = data.get("o", [])
#             highs  = data.get("h", [])
#             lows   = data.get("l", [])
#             times  = data.get("t", [])
#             if status == "no_data":
#                 return "No candle data available for this period."
#             lines = [f"Status: {status}", f"Candles returned: {len(closes)}"]
#             if closes:
#                 lines.append(f"Latest close:  {closes[-1]}")
#                 lines.append(f"Latest open:   {opens[-1] if opens else 'N/A'}")
#                 lines.append(f"Latest high:   {highs[-1] if highs else 'N/A'}")
#                 lines.append(f"Latest low:    {lows[-1] if lows else 'N/A'}")
#                 lines.append(f"Price range (all): {min(lows or closes):.2f} – {max(highs or closes):.2f}")
#             return "\n".join(lines)

#     # ── Fallback: pretty-print JSON ───────────
#     return json.dumps(data, indent=2)


# # ─────────────────────────────────────────────
# # MCP TOOL MANAGER
# # ─────────────────────────────────────────────

# class MCPToolManager:
#     """
#     Async context manager that:
#       1. Spawns the MCP server subprocess (stdio transport)
#       2. Discovers available tools
#       3. Wraps each tool as a LangChain StructuredTool
#     """

#     def __init__(self, server_script: str):
#         self.server_script = server_script
#         self.session: Optional[ClientSession] = None
#         self._exit_stack = AsyncExitStack()
#         self.langchain_tools: list[StructuredTool] = []

#     # ── context manager protocol ──────────────

#     async def __aenter__(self) -> "MCPToolManager":
#         await self._connect()
#         return self

#     async def __aexit__(self, *_) -> None:
#         await self._exit_stack.aclose()

#     # ── internals ─────────────────────────────

#     async def _connect(self) -> None:
#         command = "python" if self.server_script.endswith(".py") else "node"
#         params = StdioServerParameters(command=command, args=[self.server_script], env=None)

#         transport = await self._exit_stack.enter_async_context(stdio_client(params))
#         stdio, write = transport

#         self.session = await self._exit_stack.enter_async_context(
#             ClientSession(stdio, write)
#         )
#         await self.session.initialize()

#         response = await self.session.list_tools()
#         self.langchain_tools = self._wrap_tools(response.tools)

#         print(f"\n✅ MCP server ready — {len(self.langchain_tools)} tools available:")
#         for t in self.langchain_tools:
#             print(f"   • {t.name}")

#     def _wrap_tools(self, mcp_tools) -> list[StructuredTool]:
#         """Convert MCP tool descriptors → LangChain StructuredTools."""
#         wrapped = []
#         for mcp_tool in mcp_tools:
#             schema = mcp_tool.inputSchema or {}
#             args_schema = _build_args_schema(f"{mcp_tool.name}Schema", schema)

#             # Factory avoids the loop-closure bug: each tool gets its own name binding
#             async_fn = self._make_tool_fn(mcp_tool.name)

#             wrapped.append(
#                 StructuredTool(
#                     name=mcp_tool.name,
#                     description=mcp_tool.description or mcp_tool.name,
#                     args_schema=args_schema,
#                     coroutine=async_fn,
#                 )
#             )
#         return wrapped

#     def _make_tool_fn(self, tool_name: str):
#         """Return an async callable that invokes the named MCP tool."""
#         session = self.session

#         async def _invoke(**kwargs) -> str:
#             try:
#                 clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
#                 result = await session.call_tool(tool_name, clean_kwargs)

#                 if not result.content:
#                     return f"[{tool_name}] returned no content."

#                 # Collect raw text from all content blocks
#                 raw_texts = [
#                     c.text
#                     for c in result.content
#                     if getattr(c, "type", "") == "text"
#                 ]
#                 raw = "\n".join(raw_texts).strip()

#                 # Try to parse and reformat JSON so the LLM gets clean structured text
#                 return _format_tool_result(tool_name, raw)

#             except Exception as exc:
#                 return f"[{tool_name}] error: {exc}"

#         _invoke.__name__ = tool_name
#         return _invoke


# # ─────────────────────────────────────────────
# # FINANCIAL AGENT  (LangGraph)
# # ─────────────────────────────────────────────

# class FinancialAgent:
#     """
#     LangGraph ReAct agent.

#     Graph topology
#     ──────────────
#     START ──► agent ──(tool_calls?)──► tools ──► agent ──► ...
#                      └──(done)──────► END
#     """

#     def __init__(self, mcp_manager: MCPToolManager, model: str = "gemini-2.5-flash"):
#         self.tools = mcp_manager.langchain_tools

#         # Bind tools to the LLM so Gemini knows their signatures
#         self.llm = ChatGoogleGenerativeAI(
#             model=model,
#             temperature=0.0,
#         ).bind_tools(self.tools)

#         self._graph = self._compile_graph()
#         self._history: list[BaseMessage] = []   # persistent across turns

#     # ── graph construction ────────────────────

#     def _compile_graph(self):
#         tool_node = ToolNode(self.tools)

#         def agent_node(state: MessagesState) -> dict:
#             messages = state["messages"]

#             # Inject system prompt at the front if it's not already there
#             if not any(isinstance(m, SystemMessage) for m in messages):
#                 messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

#             response = self.llm.invoke(messages)
#             return {"messages": [response]}

#         graph = StateGraph(MessagesState)
#         graph.add_node("agent", agent_node)
#         graph.add_node("tools", tool_node)

#         graph.add_edge(START, "agent")
#         graph.add_conditional_edges("agent", tools_condition)  # → tools | END
#         graph.add_edge("tools", "agent")

#         return graph.compile()

#     # ── public API ────────────────────────────

#     async def chat(self, user_input: str) -> str:
#         """
#         Process one user turn.
#         History is maintained internally — no need to pass it manually.
#         """
#         self._history.append(HumanMessage(content=user_input))

#         result = await self._graph.ainvoke({"messages": self._history})

#         # Sync history with what the graph produced (includes tool messages)
#         self._history = list(result["messages"])

#         # Return the last AIMessage text
#         # msg.content can be a plain str OR a list of content blocks [{"type": "text", "text": "..."}]
#         for msg in reversed(self._history):
#             if isinstance(msg, AIMessage) and msg.content:
#                 content = msg.content
#                 if isinstance(content, str):
#                     return content
#                 if isinstance(content, list):
#                     parts = [
#                         block.get("text", "")
#                         for block in content
#                         if isinstance(block, dict) and block.get("type") == "text"
#                     ]
#                     text = "\n".join(p for p in parts if p).strip()
#                     if text:
#                         return text

#         return "(no response)"

#     def reset(self) -> None:
#         """Clear conversation history."""
#         self._history = []
#         print("🔄 Conversation history cleared.")


# # ─────────────────────────────────────────────
# # CLI ENTRY POINT
# # ─────────────────────────────────────────────

# HELP_TEXT = """
# Commands:
#   quit / exit  — exit the agent
#   reset        — clear conversation history
#   help         — show this message
# """


# async def main(server_script: str = "./stock_market_server.py") -> None:
#     print("━" * 52)
#     print("  📈 Financial Agent  |  LangGraph + Gemini + MCP")
#     print("━" * 52)

#     async with MCPToolManager(server_script) as mcp:
#         agent = FinancialAgent(mcp)
#         print(HELP_TEXT)

#         while True:
#             try:
#                 user_input = input("You: ").strip()
#             except (EOFError, KeyboardInterrupt):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue

#             match user_input.lower():
#                 case "quit" | "exit":
#                     print("Goodbye!")
#                     break
#                 case "reset":
#                     agent.reset()
#                     continue
#                 case "help":
#                     print(HELP_TEXT)
#                     continue

#             try:
#                 response = await agent.chat(user_input)
#                 print(f"\nAgent: {response}\n")
#             except Exception as exc:
#                 print(f"\n❌ Error: {exc}\n")


# if __name__ == "__main__":
#     asyncio.run(main())



"""
Financial Agent — LangGraph + Groq (llama-3.3-70b-versatile) + MCP
=====================================================
Architecture:
  START → [agent node] ──(has tool calls?)──► [tools node] → [agent node] → ...
                         └──(no tool calls)──► END

The agent maintains full conversation history across turns for context.
"""

import asyncio
import json
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, create_model
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMAS
# ─────────────────────────────────────────────

class ArticleSummary(BaseModel):
    headline: str = Field(..., description="Article headline")
    source: str = Field(..., description="Publisher/source name")
    date: str = Field(..., description="YYYY-MM-DD date or 'N/A'")
    link: str = Field(..., description="Source URL or 'N/A'")
    summary: str = Field(..., description="1-3 sentence summary tailored to the query")
    relevance: str = Field(..., description="Short clause explaining why this article is relevant to the query")

class UnifiedAgentResponse(BaseModel):
    type: str = Field(..., description="Response type: 'news', 'financial', or 'general'.")
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="Synthesized answer that directly addresses the query")
    top_articles: list[ArticleSummary] = Field(default=[], description="Top most relevant articles. Keep empty if type is 'financial' or 'general'.")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analysis assistant with access to real-time market data, news, and the internet.

## How you work
1. Receive a user query
2. Fetch relevant data using tools if needed (prices, news, articles, financials, or web_search)
3. READ and REASON over what the tools return (or use your own knowledge if no tools were needed)
4. Answer the query in your own words — synthesized, clear, and direct

## Output rules
- NEVER dump raw tool output or JSON at the user. Always synthesize it into a proper answer.
- For news/article queries: fetch the articles, read their content, then write a summary that directly answers the question. Cite sources by headline or publication — not raw URLs.
- For price/financial queries: present numbers cleanly (e.g. "$182.34", "P/E: 28.4x") with a brief interpretation.
- Use bullet points or short paragraphs — never walls of raw data.
- Always ground your answer in what the tools returned (if used). Never hallucinate facts.
- If it's a general question or greeting (e.g. "hi"), answer it nicely and naturally.

## News/article workflow
When a user asks something that requires news or article content:
1. Call get_market_news or get_company_news to get article list
2. Call scrape_article on the most relevant articles (top 2–3)
3. Synthesize the scraped content into a direct answer to the query
4. Format your response like this:

**[Your synthesized answer here — 2–4 sentences directly answering the query]**

📰 Sources:
- Headline 1 (Source: publication, Link: url) — brief 1-line takeaway
- Headline 2 (Source: publication, Link: url) — brief 1-line takeaway

## Other rules
- For ambiguous tickers (e.g. "Apple"), resolve to symbol (AAPL) first
- If a tool fails, try an alternative — don't give up after one error
- If the required information is not available via MCP, try searching via web_search.
"""

# ─────────────────────────────────────────────
# SCHEMA UTILS
# ─────────────────────────────────────────────

_JSON_TO_PYTHON: dict[str, type] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}


def _build_args_schema(model_name: str, json_schema: dict) -> type[BaseModel]:
    """
    Dynamically create a Pydantic v2 model from a JSON Schema dict.
    Only handles flat / one-level-deep schemas — sufficient for most MCP tools.
    """
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


# ─────────────────────────────────────────────
# TOOL RESULT FORMATTER
# ─────────────────────────────────────────────

def _format_tool_result(tool_name: str, raw: str) -> str:
    """
    Parse tool output (often JSON) and reformat it into clean readable text
    so the LLM can reason over it instead of getting raw dumps.
    """
    import json

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Not JSON — return as-is (e.g. scraped article text)
        return raw.strip()

    # ── News / article lists ──────────────────
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

    # ── Scraped article content ───────────────
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
        # Sometimes scrape returns plain text
        return str(data)

    # ── Stock price ───────────────────────────
    if tool_name == "get_stock_price":
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)

    # ── Basic financials ──────────────────────
    if tool_name == "get_basic_financials":
        if isinstance(data, dict):
            metric = data.get("metric", data)
            lines = []
            for k, v in (metric.items() if isinstance(metric, dict) else {}.items()):
                if v is not None:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines) if lines else raw

    # ── Candles (OHLCV) ──────────────────────
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

    # ── Fallback: pretty-print JSON ───────────
    return json.dumps(data, indent=2)


# ─────────────────────────────────────────────
# MCP TOOL MANAGER
# ─────────────────────────────────────────────

class MCPToolManager:
    """
    Async context manager that:
      1. Spawns the MCP server subprocess (stdio transport)
      2. Discovers available tools
      3. Wraps each tool as a LangChain StructuredTool
    """

    def __init__(self, command: str, args: list[str], env: Optional[dict] = None):
        self.command = command
        self.args = args
        self.env = env
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self.langchain_tools: list[StructuredTool] = []

    # ── context manager protocol ──────────────

    async def __aenter__(self) -> "MCPToolManager":
        await self._connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self._exit_stack.aclose()

    # ── internals ─────────────────────────────

    async def _connect(self) -> None:
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)

        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        stdio, write = transport

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self.langchain_tools = self._wrap_tools(response.tools)

        print(f"\n✅ MCP server ready — {len(self.langchain_tools)} tools available:")
        for t in self.langchain_tools:
            print(f"   • {t.name}")

    def _wrap_tools(self, mcp_tools) -> list[StructuredTool]:
        """Convert MCP tool descriptors → LangChain StructuredTools."""
        wrapped = []
        for mcp_tool in mcp_tools:
            schema = mcp_tool.inputSchema or {}
            args_schema = _build_args_schema(f"{mcp_tool.name}Schema", schema)

            # Factory avoids the loop-closure bug: each tool gets its own name binding
            async_fn = self._make_tool_fn(mcp_tool.name)

            wrapped.append(
                StructuredTool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or mcp_tool.name,
                    args_schema=args_schema,
                    coroutine=async_fn,
                )
            )
        return wrapped

    def _make_tool_fn(self, tool_name: str):
        """Return an async callable that invokes the named MCP tool."""
        session = self.session

        async def _invoke(**kwargs) -> str:
            try:
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                result = await session.call_tool(tool_name, clean_kwargs)

                if not result.content:
                    return f"[{tool_name}] returned no content."

                # Collect raw text from all content blocks
                raw_texts = [
                    c.text
                    for c in result.content
                    if getattr(c, "type", "") == "text"
                ]
                raw = "\n".join(raw_texts).strip()

                # Try to parse and reformat JSON so the LLM gets clean structured text
                return _format_tool_result(tool_name, raw)

            except Exception as exc:
                return f"[{tool_name}] error: {exc}"

        _invoke.__name__ = tool_name
        return _invoke


# ─────────────────────────────────────────────
# FINANCIAL AGENT  (LangGraph)
# ─────────────────────────────────────────────

# The LLM decides tool usage intrinsically. No hardcoded tool intercepts needed.


class FinancialAgent:
    """
    LangGraph ReAct agent.

    Graph topology
    ──────────────
    START ──► agent ──(tool_calls?)──► tools ──► agent ──► ...
                     └──(done)──────► END
    """

    def __init__(self, tools: list[StructuredTool], model: str = "gemini-2.5-flash"):
        self.tools = tools
        base_llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)
        # base_llm = ChatOpenRouter(
        #     model="qwen/qwen3.6-plus-preview:free",
        #     temperature=0.0,
        #     max_retries=2,
        #     # other params...
        # )

        # Tool-calling LLM for the agent graph
        self.llm = base_llm.bind_tools(self.tools)

        # Separate structured-output LLM for unified extraction (no tools bound)
        self.structured_llm = base_llm.with_structured_output(UnifiedAgentResponse)

        self._graph = self._compile_graph()
        self._history: list[BaseMessage] = []

    # ── graph construction ────────────────────

    def _compile_graph(self):
        tool_node = ToolNode(self.tools)

        def agent_node(state: MessagesState) -> dict:
            messages = state["messages"]

            # Inject system prompt at the front if it's not already there
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

            response = self.llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)  # → tools | END
        graph.add_edge("tools", "agent")

        return graph.compile()

    # ── public API ────────────────────────────

    async def chat(self, user_input: str) -> dict:
        """
        Process one user turn. The LLM intrinsically decides the response type.
        """
        self._history.append(HumanMessage(content=user_input))
        result = await self._graph.ainvoke({"messages": self._history})
        self._history = list(result["messages"])

        try:
            extraction_prompt = (
                f"The user asked: \"{user_input}\"\n\n"
                "Use your own knowledge or the data returned by the tools in this conversation "
                "to fill out the UnifiedAgentResponse schema. "
                "The LLM MUST decide if the context is 'news' (fill articles), 'financial' (live numbers), "
                "or 'general' (everything else, including normal chat or web searches)."
            )
            messages_for_extraction = [
                SystemMessage(content=SYSTEM_PROMPT),
                *self._history,
                HumanMessage(content=extraction_prompt),
            ]
            structured: UnifiedAgentResponse = await self.structured_llm.ainvoke(messages_for_extraction)
            return structured.model_dump()
        except Exception as exc:
            print(f"[structured extraction failed: {exc}] — falling back")
            return {"type": "financial", "query": user_input, "answer": self._extract_text()}

    def _extract_text(self) -> str:
        """Pull plain text from the last AIMessage in history."""
        for msg in reversed(self._history):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    text = "\n".join(p for p in parts if p).strip()
                    if text:
                        return text
        return "(no response)"

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []
        print("🔄 Conversation history cleared.")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

HELP_TEXT = """
Commands:
  quit / exit  — exit the agent
  reset        — clear conversation history
  help         — show this message
"""


import os

async def main(server_script: str = "./stock_market_server.py") -> None:
    print("━" * 52)
    print("  📈 Financial Agent  |  LangGraph + Groq + MCP")
    print("━" * 52)

    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_api_key:
        print("⚠️ ALPHA_VANTAGE_API_KEY not found in .env, alpha vantage tools will not be available.")

    python_cmd = "python" if server_script.endswith(".py") else "node"

    mcp1 = MCPToolManager(command=python_cmd, args=[server_script], env=None)
    mcp2 = None
    if av_api_key:
        mcp2 = MCPToolManager(command="uvx", args=["--from", "marketdata-mcp-server", "marketdata-mcp", av_api_key], env=None)

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp1)
        if mcp2:
            await stack.enter_async_context(mcp2)

        all_tools = mcp1.langchain_tools + (mcp2.langchain_tools if mcp2 else [])
        agent = FinancialAgent(tools=all_tools)
        print(HELP_TEXT)

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            match user_input.lower():
                case "quit" | "exit":
                    print("Goodbye!")
                    break
                case "reset":
                    agent.reset()
                    continue
                case "help":
                    print(HELP_TEXT)
                    continue

            try:
                result = await agent.chat(user_input)
                print("\n" + json.dumps(result, indent=2) + "\n")
            except Exception as exc:
                print(f"\n❌ Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())