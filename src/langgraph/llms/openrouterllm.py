from langchain_openai import ChatOpenAI
from src.langgraph.ui.streamlitui.loadui import LoadStreamlit
import streamlit as st
import os


class OpenrouterLLM(LoadStreamlit):
    """
    Wrapper class for initializing and managing an OpenRouter LLM model
    using LangChain's ChatOpenAI integration.
    """

    def __init__(self, user_control_input: dict):
        """
        Initialize the OpenrouterLLM class with user inputs.

        Args:
            user_control_input (dict): Dictionary containing user-provided settings:
                - OPENROUTER_API_KEY: API key string for authenticating with OpenRouter.
                - selected_openrouter_model: Model name to load from OpenRouter.
        """
        self.user_control_input = user_control_input

    def get_llm_model(self):
        """
        Initializes and returns an OpenRouter LLM instance.

        Workflow:
        - Reads the API key and model name from user_control_input.
        - If no API key is provided, warns the user in the UI.
        - Creates and returns a ChatOpenAI instance with the OpenRouter endpoint.

        Returns:
            ChatOpenAI: Configured LLM model instance, or None if inputs are missing.

        Raises:
            ValueError: If initialization fails due to invalid inputs.
        """
        try:
            openrouter_api = self.user_control_input.get("OPENROUTER_API_KEY", "")
            selected_openrouter_model = self.user_control_input.get("selected_openrouter_model", "")

            if not openrouter_api and not os.environ.get("OPENROUTER_API_KEY", ""):
                st.warning("⚠️ Please provide your **OpenRouter API Key** to continue.")
                return None

            if not selected_openrouter_model:
                st.warning("⚠️ Please select an **OpenRouter model** to proceed.")
                return None

            # Initialize the OpenRouter LLM
            llm = ChatOpenAI(
                model=selected_openrouter_model,
                api_key=openrouter_api,
                base_url="https://openrouter.ai/api/v1"
            )
            return llm

        except Exception as e:
            raise ValueError(f"❌ Failed to initialize OpenRouter LLM: {e}")
