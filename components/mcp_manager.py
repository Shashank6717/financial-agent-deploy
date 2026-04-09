import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool

from components.utils import _build_args_schema, _format_tool_result

class MCPToolManager:
    """
    Async context manager that:
      1. Spawns the MCP server subprocess (stdio transport)
      2. Discovers available tools
      3. Wraps each tool as a LangChain StructuredTool
    """

    def __init__(self, server_script: str = None, command: str = None, args: list[str] = None):
        if not server_script and not command:
            raise ValueError("Must provide either server_script or command")
        self.server_script = server_script
        self.command = command
        self.args = args or []
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self.langchain_tools: list[StructuredTool] = []

    async def __aenter__(self) -> "MCPToolManager":
        await self._connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self._exit_stack.aclose()

    async def _connect(self) -> None:
        # Pass parent environment (including API keys) to subprocess
        env = os.environ.copy()

        if self.server_script:
            cmd = "python" if self.server_script.endswith(".py") else "node"
            params = StdioServerParameters(command=cmd, args=[self.server_script], env=env)
        else:
            params = StdioServerParameters(command=self.command, args=self.args, env=env)

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
