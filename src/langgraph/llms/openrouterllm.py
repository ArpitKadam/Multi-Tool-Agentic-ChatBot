from langchain_openai import ChatOpenAI
from src.langgraph.ui.streamlitui.loadui import LoadStreamlit
import streamlit as st
import os

class OpenrouterLLM(LoadStreamlit):
    def __init__(self, user_control_input):
        self.user_control_input = user_control_input

    def get_llm_model(self):
        try:
            openrouter_api = self.user_control_input['OPENROUTER_API_KEY']
            selected_openrouter_model = self.user_control_input['selected_openrouter_model']
            if openrouter_api=='' and os.environ['OPENROUTER_API_KEY']=='':
                st.warning("☠️ Please enter your Openrouter API key")
                st.stop()
            
            llm = ChatOpenAI(model=selected_openrouter_model, api_key=openrouter_api, base_url="https://openrouter.ai/api/v1")
            return llm
        
        except Exception as e:
            raise ValueError(f"Failed to initialize Openrouter LLM: {e}")
