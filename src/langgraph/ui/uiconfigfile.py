from configparser import ConfigParser

class Config:
    def __init__(self, config_file="./src/langgraph/ui/uiconfigfile.ini"):
        self.config_file = config_file
        self.config = ConfigParser()
        self.config.read(self.config_file)

    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USE_CASE_OPTIONS").split(", ")
    
    def get_openrouter_llm_models(self):
        return self.config["DEFAULT"].get("OPENROUTER_MODEL_OPTIONS").split(", ")
    
    def get_groq_llm_models(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    def get_nvidia_llm_models(self):
        return self.config["DEFAULT"].get("NVIDIA_MODEL_OPTIONS").split(", ")
    
    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")