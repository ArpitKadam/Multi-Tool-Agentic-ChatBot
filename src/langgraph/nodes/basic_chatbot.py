from langchain_core.language_models import BaseLanguageModel
from src.langgraph.state.state import State


class BasicChatBotNode:
    """A stateless node that processes conversation history through an LLM."""

    def __init__(self, model: BaseLanguageModel):
        """
        Initializes the node with a language model.

        Args:
            model (BaseLanguageModel): An instance of a LangChain compatible language model.
        """
        if not model:
            raise ValueError("A language model instance must be provided.")
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Invokes the language model with the current conversation messages.

        Args:
            state (State): The current graph state, containing the list of messages.

        Returns:
            dict: A dictionary with the model's response message to update the state.

        Raises:
            ValueError: If the language model fails to generate a response.
        """
        try:
            # Get the list of messages from the current state
            messages = state.get("messages", [])
            if not messages:
                # Handle cases where the input might be empty
                return {"messages": []}

            # Invoke the LLM with the conversation history
            response = self.llm.invoke(messages)
            
            # Return the response in a format that updates the 'messages' key in the state
            return {"messages": [response]}
        
        except Exception as e:
            # Wrap the original exception for better error diagnosis
            raise ValueError(f"Failed to process chatbot response: {e}") from e