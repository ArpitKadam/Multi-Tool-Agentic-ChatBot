from langgraph.graph import StateGraph, START, END
from src.langgraph.state.state import State
from src.langgraph.nodes.basic_chatbot import BasicChatBotNode

class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)
        self.basic_chatbot_node = BasicChatBotNode(self.llm)

    def basic_chatbot_graph_bilder(self):
        try:
            """
            Builds a basic chatbot graph using Langgraph.
            This method initializes a chatbot node with the 'BasicChatbotNode' class and integrates into the graph.
            """
            self.graph_builder.add_node("ChatBot", self.basic_chatbot_node.process)
            self.graph_builder.add_edge(START, "ChatBot")
            self.graph_builder.add_edge("ChatBot", END)

            graph_builder = self.graph_builder.compile()
            
            return graph_builder
        
        except Exception as e:
            print(f"Error building graph: {e}")
            return None
        
    def setup_graph(self, usecase):
        try:
            """Sets up the graph for selected use case."""

            if usecase == "Basic ChatBot":
                self.graph_builder = self.basic_chatbot_graph_bilder()
                return self.graph_builder

        except Exception as e:
            print(f"Error setting up graph: {e}")
            return None