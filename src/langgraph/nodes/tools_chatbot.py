from __future__ import annotations

from typing import Callable, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.langgraph.state.state import State


class ChatBotwithToolsNode:
    """
    A factory for creating a LangGraph node that augments an LLM with tools.

    This class doesn't act as a node itself. Instead, its `process` method
    configures an LLM with a given set of tools and returns a callable function
    that can be used as a node in a StateGraph.
    """

    def __init__(self, model: BaseLanguageModel):
        """
        Initializes the node factory with a language model.

        Args:
            model (BaseLanguageModel): An instance of a LangChain compatible language model.
        """
        if not model:
            raise ValueError("A language model instance must be provided.")
        self.llm = model

    def process(self, tools: List[BaseTool]) -> Callable[[State], dict]:
        """
        Configures the LLM with tools and returns a callable node function.

        This method takes a list of tools, binds them to the initialized LLM,
        and then returns a new function (`chatbot_node`) that will execute the
        LLM with the tools when called by the graph.

        Args:
            tools (List[BaseTool]): A list of LangChain tool instances to be
                                     made available to the LLM.

        Returns:
            Callable[[State], dict]: A function that can be added as a node to a
                                     LangGraph, which processes the conversation
                                     state and may decide to call a tool.
        
        Raises:
            ValueError: If no tools are provided or if binding them to the LLM fails.
        """
        if not tools:
            raise ValueError("A list of tools must be provided to this node.")

        try:
            # Bind the tools to the LLM, creating a new model instance with tool-calling capabilities
            llm_with_tools = self.llm.bind_tools(tools)
        except Exception as e:
            raise ValueError(f"Failed to bind tools to the language model: {e}") from e

        def chatbot_node(state: State) -> dict:
            """
            The actual node logic that gets executed by the graph.
            
            It invokes the tool-augmented LLM with the current conversation state.
            """
            try:
                messages = state.get("messages", [])
                if not messages:
                    return {"messages": []}
                
                # The response may be a text message or a tool call request
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            
            except Exception as e:
                raise ValueError(f"Failed to process chatbot response with tools: {e}") from e

        return chatbot_node