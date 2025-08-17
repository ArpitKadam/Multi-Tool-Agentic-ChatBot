from src.langgraph.state.state import State

class BasicChatBotNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Represents the entry point for the basic chatbot node.
        This method processes the incoming state, generates a response using the language model,
        and updates the state with the new message.
        """
        response = self.llm.invoke(state["messages"])
        return {"messages": response}
