from src.langgraph.state.state import State


class BasicChatBotNode:
    """
    Represents a basic chatbot node that interacts with a provided LLM model.
    It processes incoming conversation state and generates responses.
    """

    def __init__(self, model):
        """
        Initialize the BasicChatBotNode with a language model.

        Args:
            model: An LLM instance (e.g., ChatGroq, ChatNVIDIA, ChatOpenAI).
        """
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Entry point for processing chatbot messages.

        Args:
            state (State): Current conversation state containing messages.

        Returns:
            dict: Updated state with the model's response included in "messages".

        Raises:
            ValueError: If model invocation fails or llm is not initialized.
        """
        if not self.llm:
            raise ValueError("❌ No LLM model provided to BasicChatBotNode.")

        try:
            response = self.llm.invoke(state.get("messages", []))
            return {"messages": response}
        except Exception as e:
            raise ValueError(f"❌ Failed to process chatbot response: {e}")
