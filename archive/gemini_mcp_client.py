import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def map_jsonschema_to_gemini(schema):
    """Maps standard JSON schema to Gemini's expected types."""
    if not isinstance(schema, dict):
        return schema
    
    mapped_schema = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            mapped_schema[key] = value.upper()
        elif isinstance(value, dict):
            mapped_schema[key] = map_jsonschema_to_gemini(value)
        elif isinstance(value, list) and key not in ["required", "enum"]:
            mapped_schema[key] = [map_jsonschema_to_gemini(v) for v in value]
        else:
            mapped_schema[key] = value
            
    return mapped_schema

class GeminiMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = genai.Client() # Picks up GEMINI_API_KEY
        self.chat = None

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
        # Configure Gemini chat session
        function_declarations = []
        for tool in tools:
            # Map the properties recursively
            properties = {}
            if tool.inputSchema and "properties" in tool.inputSchema:
                 properties = map_jsonschema_to_gemini(tool.inputSchema["properties"])
                 
            # Note: Gemini requires the root parameters object type to be set
            parameters = {
                "type": "OBJECT",
                "properties": properties,
                "required": tool.inputSchema.get("required", []) if tool.inputSchema else []
            }
            
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            })
            
        self.gemini_tools = [{"function_declarations": function_declarations}]
        
        # Start the chat session
        self.chat = self.client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=self.gemini_tools,
                temperature=0.0
            )
        )

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini 2.5 Flash and available tools"""
        response = self.chat.send_message(query)
        final_text = []

        while True:
            # If the response contains function calls
            if response.function_calls:
                for function_call in response.function_calls:
                    tool_name = function_call.name
                    # Parse the struct to a dict
                    tool_args = {}
                    if function_call.args:
                        # Sometimes function_call.args is a protobuf Struct, sometimes dict
                        tool_args = dict(function_call.args)
                    
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    
                    try:
                        # Execute tool call
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Prepare the function response for Gemini
                        # We must send exactly back what we received
                        if not result.content:
                            result_data = {"status": "success", "data": "No content returned"}
                        else:
                            # Assume text content for simple tools
                            result_data = {"output": [c.text for c in result.content if getattr(c, 'type', '') == 'text'] or getattr(result, "content", "success")}
                    except Exception as e:
                        result_data = {"error": str(e)}

                    # Send the tool result back to Gemini
                    response = self.chat.send_message(
                        # Types require proper part construction in standard approaches, but google-genai v1+ handles it simply:
                        types.Part.from_function_response(
                            name=tool_name,
                            response=result_data
                        )
                    )
            else:
                # No more function calls, we have the final text
                if response.text:
                    final_text.append(response.text)
                break

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started (Powered by Gemini)!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                elif not query:
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = GeminiMCPClient()
    try:
        await client.connect_to_server("./stock_market_server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
