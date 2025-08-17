from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
import os


class NvidiaLLM:
    """
    Wrapper class for initializing and managing an NVIDIA LLM model
    using LangChain's ChatNVIDIA integration.
    """

    def __init__(self, user_control_input: dict):
        """
        Initialize the NvidiaLLM class with user inputs.

        Args:
            user_control_input (dict): Dictionary containing user-provided settings:
                - NVIDIA_API_KEY: API key string for authenticating with NVIDIA.
                - selected_nvidia_model: Model name to load from NVIDIA.
        """
        self.user_control_input = user_control_input

    def get_llm_model(self):
        """
        Initializes and returns an NVIDIA LLM instance.

        Workflow:
        - Reads the API key and model name from user_control_input.
        - If no API key is provided, warns the user in the UI.
        - Creates and returns a ChatNVIDIA instance.

        Returns:
            ChatNVIDIA: Configured LLM model instance.

        Raises:
            ValueError: If initialization fails due to missing/invalid inputs.
        """
        try:
            nvidia_api = self.user_control_input.get("NVIDIA_API_KEY", "")
            selected_nvidia_model = self.user_control_input.get("selected_nvidia_model", "")

            if not nvidia_api and not os.environ.get("NVIDIA_API_KEY", ""):
                st.warning("⚠️ Please provide your **NVIDIA API Key** to continue.")
                return None

            if not selected_nvidia_model:
                st.warning("⚠️ Please select an **NVIDIA model** to proceed.")
                return None

            # Initialize the NVIDIA LLM
            llm = ChatNVIDIA(model=selected_nvidia_model, nvidia_api_key=nvidia_api)
            return llm

        except Exception as e:
            raise ValueError(f"❌ Failed to initialize NVIDIA LLM: {e}")
