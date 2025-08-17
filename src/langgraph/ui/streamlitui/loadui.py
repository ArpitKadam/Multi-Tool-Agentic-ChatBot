import streamlit as st
import os
from src.langgraph.ui.uiconfigfile import Config

class LoadStreamlit:
    def __init__(self):
        self.config = Config()
        self.user_control = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title="🤖🔗" + self.config.get_page_title(), layout="wide")
        st.title(self.config.get_page_title())

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            self.user_control['selected_llm'] = st.selectbox('Select LLM', llm_options)

            if self.user_control['selected_llm'] == "Groq":
                model_options = self.config.get_groq_llm_models()
                self.user_control['selected_groq_model'] = st.selectbox('Select Model', model_options)
                self.user_control['GROQ_API_KEY'] = st.session_state['GROQ_API_KEY'] = st.text_input("Groq API Key", type="password", help="Please enter your Groq API key. Don't have refer : https://console.groq.com/keys")
                if not self.user_control['GROQ_API_KEY']:
                    st.warning("☠️ Please enter your Groq API key")
                    st.stop()
            
            if self.user_control['selected_llm'] == "Openrouter":
                model_options = self.config.get_openrouter_llm_models()
                self.user_control['selected_openrouter_model'] = st.selectbox('Select Model', model_options)
                self.user_control['OPENROUTER_API_KEY'] = st.session_state['OPENROUTER_API_KEY'] = st.text_input("Openrouter API Key", type="password", help="Please enter your Openrouter API key. Don't have refer : https://openrouter.ai/settings/keys")
                if not self.user_control['OPENROUTER_API_KEY']:
                    st.warning("☠️ Please enter your Openrouter API key")
                    st.stop()

            if self.user_control['selected_llm'] == "NVIDIA":
                model_options = self.config.get_nvidia_llm_models()
                self.user_control['selected_nvidia_model'] = st.selectbox('Select Model', model_options)
                self.user_control['NVIDIA_API_KEY'] = st.session_state['NVIDIA_API_KEY'] = st.text_input("NVIDIA API Key", type="password", help="Please enter your NVIDIA API key. Don't have refer : https://build.nvidia.com/explore/discover")
                if not self.user_control['NVIDIA_API_KEY']:
                    st.warning("☠️ Please enter your NVIDIA API key")
                    st.stop()
            
            self.user_control['selected_use_case'] = st.selectbox('Select Use Case', usecase_options)

        return self.user_control