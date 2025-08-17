from configparser import ConfigParser
from pathlib import Path


class Config:
    """
    A helper class to manage UI-related configuration from an INI file.
    Provides convenient access methods for retrieving options and settings.
    """

    def __init__(self, config_file: str = "./src/langgraph/ui/uiconfigfile.ini"):
        self.config_file = Path(config_file)
        self.config = ConfigParser()

        if not self.config_file.exists():
            raise FileNotFoundError(f"❌ Config file not found at {self.config_file.resolve()}")

        self.config.read(self.config_file)

    def _get_list(self, section: str, key: str) -> list[str]:
        """
        Retrieve a config value as a list, split by commas.
        """
        value = self.config.get(section, key, fallback="")
        return [item.strip() for item in value.split(",") if item.strip()]

    def _get_value(self, section: str, key: str) -> str:
        """
        Retrieve a single config value as a string.
        """
        return self.config.get(section, key, fallback="")

    # ---- Public API ---- #
    def get_llm_options(self) -> list[str]:
        return self._get_list("DEFAULT", "LLM_OPTIONS")

    def get_usecase_options(self) -> list[str]:
        return self._get_list("DEFAULT", "USE_CASE_OPTIONS")

    def get_openrouter_llm_models(self) -> list[str]:
        return self._get_list("DEFAULT", "OPENROUTER_MODEL_OPTIONS")

    def get_groq_llm_models(self) -> list[str]:
        return self._get_list("DEFAULT", "GROQ_MODEL_OPTIONS")

    def get_nvidia_llm_models(self) -> list[str]:
        return self._get_list("DEFAULT", "NVIDIA_MODEL_OPTIONS")

    def get_page_title(self) -> str:
        return self._get_value("DEFAULT", "PAGE_TITLE")
