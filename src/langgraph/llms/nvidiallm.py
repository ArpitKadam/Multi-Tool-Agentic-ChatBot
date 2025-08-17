from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st
import os

class NvidiaLLM():
    def __init__(self, user_control_input):
        self.user_control_input = user_control_input

    def get_llm_model(self):
        try:
            nvidia_api = self.user_control_input['NVIDIA_API_KEY']
            selected_nvidia_model = self.user_control_input['selected_nvidia_model']
            if nvidia_api=='' and os.environ['NVIDIA_API_KEY']=='':
                st.warning("☠️ Please enter your NVIDIA API key")
                st.stop()
            
            llm = ChatNVIDIA(model=selected_nvidia_model, nvidia_api_key=nvidia_api)
            return llm
        
        except Exception as e:
            raise ValueError(f"Failed to initialize NVIDIA LLM: {e}")
