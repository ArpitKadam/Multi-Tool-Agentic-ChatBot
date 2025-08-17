from langgraph.graph import StateGraph, START, END
from src.langgraph.state.state import State
from src.langgraph.nodes.basic_chatbot import BasicChatBotNode
from src.langgraph.nodes.tools_chatbot import ChatBotwithToolsNode
from src.langgraph.tools.tools import get_tools, create_tools_node
from langgraph.prebuilt import tools_condition


class GraphBuilder:
    """
    A utility class for constructing LangGraph-based conversational agents.

    This class can generate two types of conversational graphs:
    1. Basic ChatBot → A simple chatbot that processes messages with an LLM.
    2. ChatBot with Tools → A tool-augmented chatbot that can invoke external tools
       (e.g., search, APIs) conditionally, based on conversation state.
    """

    def __init__(self, model):
        """
        Initialize the GraphBuilder with a specific language model (LLM).

        Args:
            model: The language model instance (e.g., ChatGroq, ChatOpenAI, ChatNVIDIA).
        """
        self.llm = model
        self.basic_chatbot_node = BasicChatBotNode(self.llm)
        self.chatbot_with_tools_node = ChatBotwithToolsNode(self.llm)

    def basic_chatbot_graph_bilder(self):
        """
        Build a **basic chatbot graph**.

        This graph has the following flow:
        START → ChatBot → END

        - "ChatBot" node uses the BasicChatBotNode to process user input.
        - Suitable for simple, single-turn or multi-turn conversations without tools.

        Returns:
            Compiled LangGraph object, or None if an error occurs.
        """
        try:
            graph_builder = StateGraph(State)

            # Add a single chatbot node
            graph_builder.add_node("ChatBot", self.basic_chatbot_node.process)

            # Define graph flow
            graph_builder.add_edge(START, "ChatBot")
            graph_builder.add_edge("ChatBot", END)

            return graph_builder.compile()

        except Exception as e:
            print(f"Error building basic chatbot graph: {e}")
            return None

    def chatbot_with_tools_graph_bilder(self):
        """
        Build a **chatbot-with-tools graph**.

        This graph has the following flow:
        START → ChatBot ↔ tools → ChatBot → END

        - "ChatBot" node uses ChatBotwithToolsNode to process input and decide if tools are needed.
        - "tools" node executes external tools (search, APIs, etc.).
        - Conditional edges (`tools_condition`) route execution to tools only when required.

        Returns:
            Compiled LangGraph object, or None if an error occurs.
        """
        try:
            graph_builder = StateGraph(State)

            # Initialize tools
            tools = get_tools()
            tool_node = create_tools_node(tools)

            # Add chatbot and tool nodes
            graph_builder.add_node("ChatBot", self.chatbot_with_tools_node.process(tools))
            graph_builder.add_node("tools", tool_node)  # lowercase matches tools_condition

            # Define graph flow
            graph_builder.add_edge(START, "ChatBot")
            graph_builder.add_conditional_edges("ChatBot", tools_condition)
            graph_builder.add_edge("tools", "ChatBot")

            return graph_builder.compile()

        except Exception as e:
            print(f"Error building chatbot-with-tools graph: {e}")
            return None

    def setup_graph(self, usecase: str):
        """
        Setup and return a graph based on the selected use case.

        Args:
            usecase (str): The type of chatbot graph to build.
                Options:
                    - "Basic ChatBot"
                    - "ChatBot with Tools"

        Returns:
            Compiled LangGraph object, or None if setup fails.
        """
        try:
            if usecase == "Basic ChatBot":
                return self.basic_chatbot_graph_bilder()

            elif usecase == "ChatBot with Tools":
                return self.chatbot_with_tools_graph_bilder()

            else:
                print(f"Unknown use case: {usecase}")
                return None

        except Exception as e:
            print(f"Error setting up graph: {e}")
            return None
