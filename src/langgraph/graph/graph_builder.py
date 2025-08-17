# --- Standard Library Imports ---
from typing import Optional

# --- Third-Party Imports ---
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

# --- Local Application Imports ---
from src.langgraph.state.state import State
from src.langgraph.nodes.basic_chatbot import BasicChatBotNode
from src.langgraph.nodes.tools_chatbot import ChatBotwithToolsNode
from src.langgraph.nodes.ai_news import AINewsNode
from src.langgraph.tools.tools import get_tools, create_tools_node


class GraphBuilder:
    """
    Constructs various LangGraph workflows based on a specified use case.

    This class acts as a factory for creating compiled graph objects for:
    1. A simple conversational agent.
    2. An agent augmented with external tools.
    3. A sequential pipeline for fetching and summarizing AI news.
    """

    def __init__(self, model: BaseLanguageModel):
        """
        Initializes the GraphBuilder with a language model and node handlers.

        Args:
            model (BaseLanguageModel): The language model instance to be used by the nodes.
        """
        self.llm = model
        self.basic_chatbot_node = BasicChatBotNode(self.llm)
        self.chatbot_with_tools_node = ChatBotwithToolsNode(self.llm)
        self.ai_news_node = AINewsNode(self.llm)

    def _build_basic_chatbot_graph(self):
        """
        Builds a graph for a basic chatbot with no external tools.

        ## Graph Flow
        `START` → `ChatBot` → `END`
        """
        graph_builder = StateGraph(State)
        graph_builder.add_node("ChatBot", self.basic_chatbot_node.process)
        graph_builder.add_edge(START, "ChatBot")
        graph_builder.add_edge("ChatBot", END)
        return graph_builder.compile()

    def _build_chatbot_with_tools_graph(self):
        """
        Builds a graph that can use tools to answer questions.

        ## Graph Flow
        `START` → `ChatBot` → (conditional) ↴
                  ↑└ `tools` ←┘
        """
        graph_builder = StateGraph(State)
        tools = get_tools()
        tool_node = create_tools_node(tools)

        # The 'ChatBot' node can either respond directly or call a tool
        graph_builder.add_node("ChatBot", self.chatbot_with_tools_node.process(tools))
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "ChatBot")
        graph_builder.add_conditional_edges("ChatBot", tools_condition)
        graph_builder.add_edge("tools", "ChatBot")
        return graph_builder.compile()

    def _build_ai_news_graph(self):
        """
        Builds a sequential graph for the AI news fetching pipeline.

        ## Graph Flow
        `START` → `FetchNews` → `Summarize` → `SaveResult` → `END`
        """
        graph_builder = StateGraph(State)
        graph_builder.add_node("FetchNews", self.ai_news_node.fetch_news)
        graph_builder.add_node("Summarize", self.ai_news_node.summarize_news)
        graph_builder.add_node("SaveResult", self.ai_news_node.save_result)

        graph_builder.add_edge(START, "FetchNews")
        graph_builder.add_edge("FetchNews", "Summarize")
        graph_builder.add_edge("Summarize", "SaveResult")
        graph_builder.add_edge("SaveResult", END)
        return graph_builder.compile()

    def setup_graph(self, usecase: str):
        """
        Selects and builds the appropriate graph based on the chosen use case.

        Args:
            usecase (str): The desired graph type. Must be one of:
                           "Basic ChatBot", "ChatBot with Tools", or "AI News".

        Returns:
            Optional[CompiledGraph]: The compiled LangGraph object, or None if the
                                     use case is invalid or an error occurs.
        """
        try:
            if usecase == "Basic ChatBot":
                return self._build_basic_chatbot_graph()
            elif usecase == "ChatBot with Tools":
                return self._build_chatbot_with_tools_graph()
            elif usecase == "AI News":
                return self._build_ai_news_graph()
            else:
                print(f"Error: Unknown use case '{usecase}'")
                return None
        except Exception as e:
            print(f"Error building graph for use case '{usecase}': {e}")
            return None