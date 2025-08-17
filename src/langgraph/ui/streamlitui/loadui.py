import streamlit as st
import os
from src.langgraph.ui.uiconfigfile import Config


class LoadStreamlit:
    """
    Loads and configures the Streamlit UI for selecting LLMs, models, and use cases.
    """
    def __init__(self):
        self.config = Config()
        self.user_control = {}

    def load_streamlit_ui(self):
        # Page setup
        st.set_page_config(
            page_title=f"🤖🔗 {self.config.get_page_title()}",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title(self.config.get_page_title())

        with st.sidebar:
            st.header("⚙️ Configuration")

            # Dropdowns
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            self.user_control['selected_llm'] = st.selectbox('🔍 Select LLM', llm_options)

            # LLM-specific configurations
            if self.user_control['selected_llm'] == "Groq":
                self._llm_section(
                    "Groq",
                    self.config.get_groq_llm_models(),
                    "GROQ_API_KEY",
                    "https://console.groq.com/keys"
                )

            elif self.user_control['selected_llm'] == "Openrouter":
                self._llm_section(
                    "Openrouter",
                    self.config.get_openrouter_llm_models(),
                    "OPENROUTER_API_KEY",
                    "https://openrouter.ai/settings/keys"
                )

            elif self.user_control['selected_llm'] == "NVIDIA":
                self._llm_section(
                    "NVIDIA",
                    self.config.get_nvidia_llm_models(),
                    "NVIDIA_API_KEY",
                    "https://build.nvidia.com/explore/discover"
                )

            # Use case selection
            self.user_control['selected_use_case'] = st.selectbox('💡 Select Use Case', usecase_options)
            st.info(
                    """
                    ### ℹ️ Chatbot with Tools – Important Info

                    ✅ **Provide your tool name** when using tools.  
                    🔧 **Supported Tools**:  
                    - `ARXIV_TOOL`  
                    - `WIKI_TOOL`  
                    - `DUCK_TOOL`  
                    - `TAVILY_TOOL`  
                    - `BRAVE_TOOL`  
                    - `GOOGLE_SCHOLAR_TOOL`  
                    - `GOOGLE_FINANCE_TOOL`  
                    - `GOOGLE_JOBS_TOOL`  
                    - `SERP_HOTEL_TOOL`  

                    ⚠️ **Model Limitation Notice**:  
                    - 🚫 *Do not use OpenRouter models* for tool execution  
                    - ✅ Instead, use only **Groq** models for reliable results  
                    """,
                    )

            # If chatbot with tools, load tool API keys from environment
            if self.user_control['selected_use_case'] == "Chatbot with Tools":
                st.subheader("🔑 Tool API Keys (from environment)")
                self.user_control['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
                self.user_control['BRAVE_API_KEY'] = os.getenv("BRAVE_API_KEY")
                self.user_control['SERP_API_KEY'] = os.getenv("SERP_API_KEY")

                with st.expander("View Loaded Keys", expanded=False):
                    st.write({
                        "TAVILY_API_KEY": "Loaded ✅" if self.user_control['TAVILY_API_KEY'] else "Missing ❌",
                        "BRAVE_API_KEY": "Loaded ✅" if self.user_control['BRAVE_API_KEY'] else "Missing ❌",
                        "SERP_API_KEY": "Loaded ✅" if self.user_control['SERP_API_KEY'] else "Missing ❌",
                    })

        return self.user_control

    def _llm_section(self, provider_name, model_options, api_key_name, help_url):
        """
        Helper method to avoid repetition in LLM-specific sections.
        """
        st.subheader(f"🔧 {provider_name} Settings")
        self.user_control[f'selected_{provider_name.lower()}_model'] = st.selectbox('Select Model', model_options)

        # API key input
        self.user_control[api_key_name] = st.session_state[api_key_name] = st.text_input(
            f"{provider_name} API Key",
            type="password",
            help=f"Please enter your {provider_name} API key. Get one here: {help_url}"
        )

        # Warn if missing
        if not self.user_control[api_key_name]:
            st.warning(f"☠️ Please enter your {provider_name} API key")
