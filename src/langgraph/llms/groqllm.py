from langchain_groq import ChatGroq
import streamlit as st
import os


class GroqLLM:
    """
    Wrapper class for initializing and managing a Groq LLM model
    using LangChain's ChatGroq integration.
    """

    def __init__(self, user_control_input: dict):
        """
        Initialize the GroqLLM class with user inputs.

        Args:
            user_control_input (dict): Dictionary containing user-provided settings:
                - GROQ_API_KEY: API key string for authenticating with Groq.
                - selected_groq_model: Model name to load from Groq.
        """
        self.user_control_input = user_control_input

    def get_llm_model(self):
        """
        Initializes and returns a Groq LLM instance.

        Workflow:
        - Reads the API key and model name from user_control_input.
        - If no API key is provided, warns the user in the UI.
        - Creates and returns a ChatGroq instance.

        Returns:
            ChatGroq: Configured LLM model instance.

        Raises:
            ValueError: If initialization fails due to missing/invalid inputs.
        """
        try:
            groq_api = self.user_control_input.get("GROQ_API_KEY", "")
            selected_groq_model = self.user_control_input.get("selected_groq_model", "")

            if not groq_api and not os.environ.get("GROQ_API_KEY", ""):
                st.warning("⚠️ Please provide your **Groq API Key** to continue.")
                return None

            if not selected_groq_model:
                st.warning("⚠️ Please select a **Groq model** to proceed.")
                return None

            # Initialize the Groq LLM
            llm = ChatGroq(model=selected_groq_model, api_key=groq_api)
            return llm

        except Exception as e:
            raise ValueError(f"❌ Failed to initialize Groq LLM: {e}")
