"""
Main entry point for the LangGraph Agentic ChatBot Streamlit application.

This script initializes the user interface, handles user inputs from various UI
components, sets up the appropriate LangGraph agent based on user selection,
and displays the results.
"""
import streamlit as st
from typing import Dict, Any

# Local application imports
from src.langgraph.ui.streamlitui.loadui import LoadStreamlit
from src.langgraph.llms.openrouterllm import OpenrouterLLM
from src.langgraph.llms.groqllm import GroqLLM
from src.langgraph.llms.nvidiallm import NvidiaLLM
from src.langgraph.graph.graph_builder import GraphBuilder
from src.langgraph.ui.streamlitui.display_result import DisplayResultStreamlit


def process_request(user_message: str, ui_settings: Dict[str, Any]):
    """
    Initializes the model, builds the graph, and runs the agent to process the user's request.

    This function serves as the core processing pipeline for any user input.

    Args:
        user_message (str): The message or command from the user.
        ui_settings (Dict[str, Any]): A dictionary containing settings from the UI,
                                      like the selected LLM and use case.
    """
    # A mapping from UI provider names to their respective LLM handler classes.
    llm_providers = {
        "Openrouter": OpenrouterLLM,
        "Groq": GroqLLM,
        "NVIDIA": NvidiaLLM,
    }

    try:
        # --- 1. Initialize the Language Model ---
        selected_llm_provider = ui_settings.get("selected_llm")
        llm_class = llm_providers.get(selected_llm_provider)
        if not llm_class:
            st.error(f"❌ Unsupported LLM provider: {selected_llm_provider}")
            return

        llm = llm_class(ui_settings).get_llm_model()
        usecase = ui_settings.get("selected_use_case")

    except Exception as e:
        st.error(f"⚠️ **Model Initialization Error:**\n\nCould not initialize the selected language model. Please check your API keys and model settings.\n\n*Details: {e}*")
        st.stop()

    try:
        # --- 2. Build the appropriate graph ---
        graph_builder = GraphBuilder(llm)
        graph = graph_builder.setup_graph(usecase)
        if not graph:
            st.error(f"⚠️ **Graph Building Error:**\n\nCould not build the graph for the '{usecase}' use case.")
            st.stop()

        # --- 3. Display the result on the UI ---
        DisplayResultStreamlit(
            usecase=usecase, graph=graph, user_message=user_message
        ).display_result_on_ui()

    except Exception as e:
        st.error(f"⚠️ **Application Error:**\n\nAn unexpected error occurred while processing your request.\n\n*Details: {e}*")
        st.stop()


def run_agentic_chatbot_app():
    """
    Loads and runs the LangGraph Agentic ChatBot application.

    This function sets up the Streamlit UI and manages the input flow, triggering
    the processing pipeline for either standard chat messages or specific button actions.
    """
    st.set_page_config(
        page_title="LangGraph: Multi-Agent Chat",
        page_icon="🤖",
        layout="wide"
    )

    # --- Load UI components and get user settings ---
    ui_loader = LoadStreamlit()
    ui_settings = ui_loader.load_streamlit_ui()

    if not ui_settings:
        st.error("🚨 Critical Error: Could not load UI components. The application cannot continue.")
        st.stop()

    # --- Handle user input triggers ---
    # This section checks which UI element the user interacted with.

    # Trigger 1: User clicked the "Fetch Latest News" button for the AI News agent.
    if st.session_state.get("IsFetchButtonClicked", False):
        st.session_state.IsFetchButtonClicked = False  # Reset the flag after use
        timeframe = st.session_state.get("timeframe", "daily")
        process_request(user_message=timeframe, ui_settings=ui_settings)

    # Trigger 2: User sent a message through the main chat input.
    elif prompt := st.chat_input("Ask me anything..."):
        process_request(user_message=prompt, ui_settings=ui_settings)


# To run the app, you would typically call this function from your main script entry point.
# if __name__ == "__main__":
#     run_agentic_chatbot_app()