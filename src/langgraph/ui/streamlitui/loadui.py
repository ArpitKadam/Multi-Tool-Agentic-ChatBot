import os
import streamlit as st
from typing import Dict, Any, List

from src.langgraph.ui.uiconfigfile import Config


class LoadStreamlit:
    """
    Manages the creation and state of the Streamlit user interface.

    This class handles rendering all sidebar components, including model selection,
    API key inputs, and use-case-specific controls.
    """

    def __init__(self):
        """Initializes the UI loader with a configuration object."""
        self.config = Config()
        self.user_settings: Dict[str, Any] = {}

    def load_streamlit_ui(self) -> Dict[str, Any]:
        """
        Renders the complete Streamlit UI and returns user-selected settings.

        Returns:
            Dict[str, Any]: A dictionary containing all user-configured settings.
        """
        st.set_page_config(
            page_title=f"{self.config.get_page_title()}",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title(f"🔗 {self.config.get_page_title()}")

        with st.sidebar:
            st.header("⚙️ Configuration")
            st.divider()

            self._render_model_selection()
            st.divider()

            self._render_use_case_selection()

        return self.user_settings

    def _render_model_selection(self):
        """Renders the UI components for selecting and configuring the LLM provider."""
        st.subheader("1. Select Language Model")
        llm_options = self.config.get_llm_options()
        self.user_settings['selected_llm'] = st.selectbox(
            'Select LLM Provider', llm_options, label_visibility="collapsed"
        )

        # Dynamically render settings based on the selected LLM provider
        provider = self.user_settings['selected_llm']
        if provider == "Groq":
            self._render_llm_settings(
                "Groq", self.config.get_groq_llm_models(), "GROQ_API_KEY", "https://console.groq.com/keys"
            )
        elif provider == "Openrouter":
            self._render_llm_settings(
                "Openrouter", self.config.get_openrouter_llm_models(), "OPENROUTER_API_KEY", "https://openrouter.ai/settings/keys"
            )
        elif provider == "NVIDIA":
            self._render_llm_settings(
                "NVIDIA", self.config.get_nvidia_llm_models(), "NVIDIA_API_KEY", "https://build.nvidia.com/explore/discover"
            )

    def _render_llm_settings(self, provider_name: str, model_options: List[str], api_key_name: str, help_url: str):
        """
        A generic helper to render the model selection and API key input for an LLM provider.
        """
        self.user_settings[f'selected_{provider_name.lower()}_model'] = st.selectbox(f'{provider_name} Model', model_options)
        
        api_key = st.text_input(
            f"{provider_name} API Key",
            type="password",
            help=f"Get your key from [this link]({help_url})",
        )
        self.user_settings[api_key_name] = api_key
        
        if not api_key:
            st.warning(f"Please enter your {provider_name} API key to continue.")

    def _render_use_case_selection(self):
        """Renders the UI for selecting the agent's use case and related tools."""
        st.subheader("2. Select Use Case")
        usecase_options = self.config.get_usecase_options()
        self.user_settings['selected_use_case'] = st.selectbox(
            'Select Agent Type', usecase_options, label_visibility="collapsed"
        )

        use_case = self.user_settings['selected_use_case']
        
        # Render tool configurations if the selected use case requires them
        if use_case in ["ChatBot with Tools", "AI News"]:
            self._render_tool_config(use_case)

    def _render_tool_config(self, use_case: str):
        """Renders tool-specific UI components based on the selected use case."""
        st.divider()
        st.subheader("🛠️ Tool Configuration")

        # Load API keys from environment variables
        tool_keys = ["TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERP_API_KEY"]
        for key in tool_keys:
            self.user_settings[key] = os.getenv(key)
        
        # Display the status of loaded tool keys in an expander
        with st.expander("View Loaded Tool Keys"):
            for key in tool_keys:
                status = "✅ Loaded" if self.user_settings[key] else "❌ Missing"
                st.text(f"{key}: {status}")

        # Render use-case specific controls and information
        if use_case == "AI News":
            st.markdown("##### AI News Explorer")
            timeframe = st.selectbox(
                'Select Time Frame',
                ['Daily', 'Weekly', 'Monthly', 'Yearly'],
                index=0
            )
            if st.button("🔄 Fetch Latest News", use_container_width=True, type="primary"):
                st.session_state.IsFetchButtonClicked = True
                st.session_state.timeframe = timeframe.lower()
            
            with st.expander("ℹ️ AI News - Important Info"):
                st.info(
                    """
                    - **Default Tool:** This agent exclusively uses the `TAVILY_TOOL` to fetch news.
                    - **Model Recommendation:** For best results, use **Groq** models. OpenRouter models may not be reliable for this task.
                    """
                )
        
        elif use_case == "ChatBot with Tools":
            with st.expander("ℹ️ Chatbot with Tools - Important Info"):
                st.info(
                    """
                    - **Usage:** To use a tool, please specify its name in your prompt (e.g., "Use TAVILY_TOOL to search for...").
                    - **Model Recommendation:** For reliable tool execution, use **Groq** models.
                    """
                )
                st.markdown(
                    """
                    **Supported Tools:**
                    - `ARXIV_TOOL`
                    - `WIKI_TOOL`
                    - `DUCK_TOOL`
                    - `TAVILY_TOOL`
                    - `BRAVE_TOOL`
                    - `GOOGLE_SCHOLAR_TOOL`
                    - `GOOGLE_FINANCE_TOOL`
                    - `GOOGLE_JOBS_TOOL`
                    - `SERP_HOTEL_TOOL`
                    """
                )