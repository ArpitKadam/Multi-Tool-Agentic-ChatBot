import streamlit as st
from src.langgraph.ui.streamlitui.loadui import LoadStreamlit
from src.langgraph.llms.openrouterllm import OpenrouterLLM
from src.langgraph.llms.groqllm import GroqLLM
from src.langgraph.llms.nvidiallm import NvidiaLLM
from src.langgraph.graph.graph_builder import GraphBuilder
from src.langgraph.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI app with streamlit UI.
    This function initializes the UI, handles user input and configures the llm model.
    Its sets up the graph based on selected use case and llm model.
    """

    UI = LoadStreamlit()
    user_input = UI.load_streamlit_ui()
    if not user_input:
        st.warning("☠️ Failed to load user input")
        st.stop()

    user_message = st.chat_input("Enter your message")

    if user_message:
        try:
            if user_input['selected_llm'] == "Openrouter":
                llm = OpenrouterLLM(user_input).get_llm_model()
            elif user_input['selected_llm'] == "Groq":
                llm = GroqLLM(user_input).get_llm_model()
            elif user_input['selected_llm'] == "NVIDIA":
                llm = NvidiaLLM(user_input).get_llm_model()

            usecase = user_input.get("selected_use_case")

        except Exception as e:
            st.error(f"Error processing initialization of the model: {e}")
            st.stop() 

        try:
            graph_builder = GraphBuilder(llm)
            graph = graph_builder.setup_graph(usecase)
            DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
        
        except Exception as e:
            st.error(f"Error setting up graph: {e}")
            st.stop() 

        