from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import StructuredTool

from components.utils import _build_args_schema, _format_tool_result

class MCPToolManagerSSE:
    """
    Async context manager that connects to an MCP server over SSE transport.
    Matches MCPToolManager's interface (langchain_tools list).
    """

    def __init__(self, url: str):
        self.url = url
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self.langchain_tools: list[StructuredTool] = []

    async def __aenter__(self) -> "MCPToolManagerSSE":
        await self._connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self._exit_stack.aclose()

    async def _connect(self) -> None:
        transport = await self._exit_stack.enter_async_context(sse_client(self.url))
        stdio, write = transport

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self.langchain_tools = self._wrap_tools(response.tools)

        print(f"\n✅ MCP server (SSE) ready — {len(self.langchain_tools)} tools available:")
        for t in self.langchain_tools:
            print(f"   • {t.name}")

    def _wrap_tools(self, mcp_tools) -> list[StructuredTool]:
        wrapped = []
        for mcp_tool in mcp_tools:
            schema = mcp_tool.inputSchema or {}
            args_schema = _build_args_schema(f"{mcp_tool.name}Schema", schema)

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
        session = self.session

        async def _invoke(**kwargs) -> str:
            try:
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                result = await session.call_tool(tool_name, clean_kwargs)

                if not result.content:
                    return f"[{tool_name}] returned no content."

                raw_texts = [
                    c.text
                    for c in result.content
                    if getattr(c, "type", "") == "text"
                ]
                raw = "\n".join(raw_texts).strip()
                return _format_tool_result(tool_name, raw)

            except Exception as exc:
                return f"[{tool_name}] error: {exc}"

        _invoke.__name__ = tool_name
        return _invoke
