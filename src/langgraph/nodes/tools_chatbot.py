from src.langgraph.state.state import State
from typing import Callable, List, Any, Dict


class ChatBotwithToolsNode:
    """
    Represents a chatbot node that can use tools in addition to the base LLM.
    The node binds tools to the LLM and returns a callable for processing messages.
    """

    def __init__(self, model):
        """
        Initialize the ChatBotwithToolsNode with a language model.

        Args:
            model: An LLM instance (e.g., ChatGroq, ChatNVIDIA, ChatOpenAI).
        """
        self.llm = model

    def process(self, tools: List[Any]) -> Callable[[State], Dict[str, Any]]:
        """
        Binds tools to the LLM and returns a chatbot node function.

        Args:
            tools (List[Any]): List of tools to bind with the LLM.

        Returns:
            Callable[[State], Dict[str, Any]]: Function that processes state and generates responses.

        Raises:
            ValueError: If LLM is not initialized or binding fails.
        """
        if not self.llm:
            raise ValueError("❌ No LLM model provided to ChatBotwithToolsNode.")

        if not tools:
            raise ValueError("❌ No tools provided to ChatBotwithToolsNode.")

        try:
            llm_with_tools = self.llm.bind_tools(tools)
        except Exception as e:
            raise ValueError(f"❌ Failed to bind tools to LLM: {e}")

        def chatbot_node(state: State) -> Dict[str, Any]:
            """
            Chat logic for processing the input state and returning the updated state.

            Args:
                state (State): Current conversation state containing messages.

            Returns:
                dict: Updated state with the model's response included in "messages".
            """
            try:
                response = llm_with_tools.invoke(state.get("messages", []))
                return {"messages": [response]}
            except Exception as e:
                raise ValueError(f"❌ Failed to process chatbot response with tools: {e}")

        return chatbot_node
