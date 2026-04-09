from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from components.mcp_manager import MCPToolManager
from components.schemas import NewsSummaryResponse
from components.utils import SYSTEM_PROMPT

_NEWS_TOOLS = {"get_market_news", "get_company_news", "scrape_article","get_stock_symbol_lookup","get_stock_price"}

class FinancialAgent:
    """
    LangGraph ReAct agent.

    Graph topology
    ──────────────
    START ──► agent ──(tool_calls?)──► tools ──► agent ──► ...
                     └──(done)──────► END
    """

    def __init__(self, mcp_managers: list[MCPToolManager], model: str = "llama-3.3-70b-versatile"):
        self.tools = []
        for mgr in mcp_managers:
            self.tools.extend(mgr.langchain_tools)
        base_llm = ChatGroq(model=model, temperature=0.0)
        # base_llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)
        self.base_llm = base_llm

        # Tool-calling LLM for the agent graph
        self.llm = base_llm.bind_tools(self.tools)

        # Separate structured-output LLM for news extraction (no tools bound)
        self.structured_llm = base_llm.with_structured_output(NewsSummaryResponse, method="json_mode")

        self._graph = self._compile_graph()
        self._history: list[BaseMessage] = []

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

    async def chat(self, user_input: str) -> dict:
        """
        Process one user turn. Always returns a dict:
          {"type": "news",      "query": ..., "answer": ..., "top_articles": [...]}
          {"type": "financial", "query": ..., "answer": ..., "top_articles": [...]}
        """

        self._history.append(HumanMessage(content=user_input))
        result = await self._graph.ainvoke({"messages": self._history})
        self._history = list(result["messages"])

        tool_names_used = {
            m.name for m in self._history if isinstance(m, ToolMessage)
        }

        if not tool_names_used:
            return {"type": "general", "query": user_input, "answer": self._extract_text(), "top_articles": []}

        news_tools_used = any(name in _NEWS_TOOLS for name in tool_names_used)

        if not news_tools_used:
            return {"type": "financial", "query": user_input, "answer": self._extract_text(), "top_articles": []}

        try:
            structured: NewsSummaryResponse = await self._extract_news_structured(user_input)
            return {
                "type": "news",
                "query": structured.query,
                "answer": structured.answer,
                "top_articles": [a.model_dump() for a in structured.top_articles],
            }
        except Exception as exc:
            print(f"[structured extraction failed: {exc}] — falling back")
            answer_text = self._extract_text()
            articles = self._extract_top_articles_from_text(answer_text)
            return {"type": "financial", "query": user_input, "answer": answer_text, "top_articles": articles}

    async def _extract_news_structured(self, user_input: str) -> NewsSummaryResponse:
        import json
        import re
        schema_json = NewsSummaryResponse.model_json_schema()
        extraction_prompt = (
            f"The user asked: \"{user_input}\"\n\n"
            "Using ONLY the article data returned by the tools in this conversation, "
            "provide a JSON object matching this exact schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            "IMPORTANT: You MUST include at least 5 articles in top_articles. Always aim for 5 or more.\n"
            "Return ONLY valid JSON. Do not include markdown tags like ```json or any trailing characters."
        )
        messages_for_extraction = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self._history,
            HumanMessage(content=extraction_prompt),
        ]
        
        # Bypass LangChain's broken structured parser for this model and do robust manual parsing
        response = await self.base_llm.ainvoke(messages_for_extraction)
        content = response.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Qwen 32B sometimes emits garbage at the end of the JSON object, fix it with regex
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
             content_clean = match.group(1)
             try:
                 data = json.loads(content_clean)
                 return NewsSummaryResponse(**data)
             except json.JSONDecodeError:
                 pass # Fall through to the final exception below
        
        raise ValueError(f"Manual JSON validation failed. Content was: {content}")

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

    def _extract_top_articles_from_text(self, text: str) -> list[dict]:
        """Attempt to extract top articles from the text response's 'Sources' section."""
        top_articles = []
        if "Sources:" in text or "Sources:\n" in text:
            import re
            parts = re.split(r'📰?\s*\**Sources?:?\**\s*\n', text)
            if len(parts) > 1:
                sources_part = parts[-1]
                lines = sources_part.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('-'):
                        top_articles.append({
                            "headline": line.lstrip('- ').strip(),
                            "source": "N/A",
                            "date": "N/A",
                            "link": "N/A",
                            "summary": "N/A",
                            "relevance": "N/A"
                        })
        return top_articles

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []
        print("🔄 Conversation history cleared.")