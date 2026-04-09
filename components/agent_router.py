from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from components.mcp_manager import MCPToolManager
from components.schemas_router import UnifiedResponse
from components.utils import SYSTEM_PROMPT

class FinancialAgentRouter:
    """
    LangGraph ReAct agent with LLM-routed response typing.

    No heuristic routing or if/else checks. The LLM decides:
      - Whether to call tools
      - Whether the final response is "news" or "financial"
      - How to structure the output
    """

    def __init__(self, mcp_managers: list[MCPToolManager], model: str = "qwen/qwen3-32b"):
        self.tools = []
        for mgr in mcp_managers:
            self.tools.extend(mgr.langchain_tools)

        base_llm = ChatGroq(model=model, temperature=0.0)
        self.base_llm = base_llm

        # Tool-calling LLM for the agent graph
        self.llm = base_llm.bind_tools(self.tools)

        self._graph = self._compile_graph()
        self._history: list[BaseMessage] = []

    def _compile_graph(self):
        tool_node = ToolNode(self.tools)

        def agent_node(state: MessagesState) -> dict:
            messages = state["messages"]
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

            response = self.llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def chat(self, user_input: str) -> dict:
        """
        Process one user turn. Always returns a dict matching UnifiedResponse.
        The model decides response type and article list.
        """
        self._history.append(HumanMessage(content=user_input))
        result = await self._graph.ainvoke({"messages": self._history})
        self._history = list(result["messages"])

        structured = await self._extract_unified_structured(user_input)
        return structured.model_dump()

    async def _extract_unified_structured(self, user_input: str) -> UnifiedResponse:
        import json
        import re

        schema_json = UnifiedResponse.model_json_schema()
        extraction_prompt = (
            f"The user asked: \"{user_input}\"\n\n"
            "Using ONLY the tool outputs in this conversation (if any), "
            "return a JSON object that matches this exact schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            "Rules:\n"
            "- If no news sources were used, set type to \"financial\" and top_articles to [].\n"
            "- If news sources were used, set type to \"news\" and include at least 5 top_articles (minimum 5, more if available).\n"
            "- Return ONLY valid JSON with no markdown fences or extra text."
        )

        messages_for_extraction = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self._history,
            HumanMessage(content=extraction_prompt),
        ]

        response = await self.base_llm.ainvoke(messages_for_extraction)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            content_clean = match.group(1)
            try:
                data = json.loads(content_clean)
                return UnifiedResponse(**data)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Manual JSON validation failed. Content was: {content}")

    def reset(self) -> None:
        self._history = []
        print("\ud83d\udd04 Conversation history cleared.")
