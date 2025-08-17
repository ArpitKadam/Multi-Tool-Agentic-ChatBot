from langchain_groq import ChatGroq
import streamlit as st
import os

class GroqLLM():
    def __init__(self, user_control_input):
        self.user_control_input = user_control_input

    def get_llm_model(self):
        try:
            groq_api = self.user_control_input['GROQ_API_KEY']
            selected_groq_model = self.user_control_input['selected_groq_model']
            if groq_api=='' and os.environ['GROQ_API_KEY']=='':
                st.warning("☠️ Please enter your Groq API key")
                st.stop()
            
            llm = ChatGroq(model=selected_groq_model, api_key=groq_api)
            return llm
        
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq LLM: {e}")
