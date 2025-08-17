import streamlit as st
from src.langgraph.ui.streamlitui.loadui import LoadStreamlit
from src.langgraph.llms.openrouterllm import OpenrouterLLM
from src.langgraph.llms.groqllm import GroqLLM
from src.langgraph.llms.nvidiallm import NvidiaLLM
from src.langgraph.graph.graph_builder import GraphBuilder
from src.langgraph.ui.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI app with Streamlit UI.
    - Initializes the UI and retrieves user input.
    - Configures the selected LLM model.
    - Builds the graph based on the selected use case.
    - Displays the result interactively.
    """

    # Load Streamlit UI
    ui_loader = LoadStreamlit()
    user_input = ui_loader.load_streamlit_ui()

    if not user_input:
        st.warning("⚠️ Failed to load user input")
        return

    user_message = st.chat_input("Enter your message")
    if not user_message:
        return

    # Map LLM providers to their respective classes
    llm_providers = {
        "Openrouter": OpenrouterLLM,
        "Groq": GroqLLM,
        "NVIDIA": NvidiaLLM,
    }

    try:
        llm_class = llm_providers.get(user_input.get("selected_llm"))
        if not llm_class:
            st.error(f"❌ Unsupported LLM provider: {user_input.get('selected_llm')}")
            return

        llm = llm_class(user_input).get_llm_model()
        usecase = user_input.get("selected_use_case")

    except Exception as e:
        st.error(f"⚠️ Error initializing model: {e}")
        st.stop()

    try:
        graph_builder = GraphBuilder(llm)
        graph = graph_builder.setup_graph(usecase)

        DisplayResultStreamlit(
            usecase=usecase, graph=graph, user_message=user_message
        ).display_result_on_ui()

    except Exception as e:
        st.error(f"⚠️ Error setting up graph: {e}")
        st.stop()
